#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <MeshField.hpp>
#include <KokkosController.hpp>
#include <span>
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <Kokkos_Profiling_ProfileSection.hpp>
#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if(argc!=3 && !lib.world()->rank()) {
    fprintf(stderr, "Usage: %s <input mesh .osh> <output vtk .vtk>\n", argv[0]);
  }
  OMEGA_H_CHECK(argc == 3);
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(argv[1], lib.world(), &mesh);
  using Ctrlr = Controller::KokkosController<MemorySpace, ExecutionSpace, int*, int**>;
  Ctrlr c({mesh.nverts(), mesh.nverts(), 2});
  Kokkos::Timer timer;
  MeshField::MeshField<Ctrlr> mf(c);

  auto vtxRankId = mf.makeField<1>();

  const int rank = lib.world()->rank();
  auto setVtx = KOKKOS_LAMBDA (const int i) {
    vtxRankId(i, 0) = rank;
    vtxRankId(i, 1) = rank;
  };
  mf.parallel_for({0},{mesh.nverts()}, setVtx, "set_vertex");
  Kokkos::Timer ask_dist_timer;
  mesh.ask_dist(0);
  //std::cout << "ask_dist Kokkos Timer: " << ask_dist_timer.seconds() << std::endl;
  lib.world()->barrier();
  //std::cout << "ask_dist Kokkos Timer: " << ask_dist_timer.seconds() << std::endl;
  Kokkos::fence();
  
  std::vector<double> serializeTimes;
  double serializeStartTime = 0;
  std::vector<double> syncTimes;
  double syncStartTime = 0;
  std::vector<double> deserializeTimes;
  double deserializeStartTime = 0;
  int numRuns = 10;

  //Use the dist to synchronize values across the vertices - the 'owner' of each
  for(int i = 0; i < numRuns; ++i){
    serializeStartTime = timer.seconds();
    auto vtxRankIdView = vtxRankId.serialize();
    serializeTimes.push_back(timer.seconds() - serializeStartTime);
    Kokkos::fence();

    Omega_h::Write<Omega_h::LO> fieldWrite(vtxRankIdView);
    int width = 2;
    syncStartTime = timer.seconds();
    auto syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), width);
    syncTimes.push_back(timer.seconds() - syncStartTime);
    Kokkos::fence();
 
    deserializeStartTime = timer.seconds();
	  vtxRankId.deserialize(syncFieldRead.view());
    deserializeTimes.push_back(timer.seconds() - deserializeStartTime);
    Kokkos::fence();
  }

  double serializeSum = 0;
  double deserializeSum = 0;
  double syncSum = 0;
  for(int i = 0; i < numRuns; ++i){
    serializeSum += serializeTimes[i];
    deserializeSum += deserializeTimes[i];
    syncSum += syncTimes[i];
  }
  
  double serializeAllSum, serializeAllMin, serializeAllMax;
  double serializeMin = *std::min_element(serializeTimes.begin(), serializeTimes.end());
  double serializeMax = *std::max_element(serializeTimes.begin(), serializeTimes.end());
  MPI_Allreduce(&serializeSum, &serializeAllSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
  MPI_Allreduce(&serializeMin, &serializeAllMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); 
  MPI_Allreduce(&serializeMax, &serializeAllMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 

  double deserializeAllSum, deserializeAllMin, deserializeAllMax;
  double deserializeMin = *std::min_element(deserializeTimes.begin(), deserializeTimes.end());
  double deserializeMax = *std::max_element(deserializeTimes.begin(), deserializeTimes.end());
  MPI_Allreduce(&deserializeSum, &deserializeAllSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
  MPI_Allreduce(&deserializeMin, &deserializeAllMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); 
  MPI_Allreduce(&deserializeMax, &deserializeAllMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 

  double syncAllSum, syncAllMin, syncAllMax;
  double syncMin = *std::min_element(syncTimes.begin(), syncTimes.end());
  double syncMax = *std::max_element(syncTimes.begin(), syncTimes.end());
  MPI_Allreduce(&syncSum, &syncAllSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
  MPI_Allreduce(&syncMin, &syncAllMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); 
  MPI_Allreduce(&syncMax, &syncAllMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 

  double syncAvg = syncAllSum / numRuns / 4;
  double serializeAvg = serializeAllSum / numRuns / 4;
  double deserializeAvg = deserializeAllSum / numRuns / 4;
  
  std::cout << "computation, " << "sync, " << "serialize, " << "deserialize" << std::endl <<
               "min, " << syncAllMin << ", " << serializeAllMin << ", " << deserializeAllMin << std::endl <<
               "max, " << syncAllMax << ", " << serializeAllMax << ", " << deserializeAllMax << std::endl <<
               "avg, " << syncAvg << ", " << serializeAvg << ", " << deserializeAvg << std::endl;

  //check the meshfield Data against the Omega_H ownership array
  auto owners = mesh.ask_owners(0).ranks;
  double error = 0;
  Kokkos::parallel_reduce ("Reduction", mesh.nverts(), KOKKOS_LAMBDA (const int i, double& update) {
	  update += abs(owners[i] - vtxRankId(i, 0));
	  update += abs(owners[i] - vtxRankId(i, 1));
  }, error);
  OMEGA_H_CHECK(error == 0);
  //convert the meshfield to an omegah 'tag' for visualization
  Omega_h::Write<Omega_h::LO> vtxVals(mesh.nverts());
  auto mfToOmegah = KOKKOS_LAMBDA (const int i) {
    vtxVals[i] = vtxRankId(i, 0);
  };
  mf.parallel_for({0},{mesh.nverts()}, mfToOmegah, "meshField_to_omegah");
  mesh.add_tag(0, "fromMeshFieldInt", 1, Omega_h::read(vtxVals));

  //write vtk files
  Omega_h::vtk::write_parallel(argv[2], &mesh, mesh.dim());
  return 0;
}
