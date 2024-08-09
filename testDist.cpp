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
#include <numeric>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

void getMinMaxAvg(MPI_Comm &comm, int worldSize, std::vector<double> &timing, double &avg, double &min, double &max) {
  double tempSum = std::reduce(timing.begin(), timing.end());
  double tempMin = *std::min_element(timing.begin(), timing.end());
  double tempMax = *std::max_element(timing.begin(), timing.end());
  double sum;
  MPI_Allreduce(&tempSum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm); 
  MPI_Allreduce(&tempMin, &min, 1, MPI_DOUBLE, MPI_MIN, comm); 
  MPI_Allreduce(&tempMax, &max, 1, MPI_DOUBLE, MPI_MAX, comm); 
  
  avg = sum / timing.size() / worldSize; 
}

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
    Kokkos::fence();
    serializeTimes.push_back(timer.seconds() - serializeStartTime);

    Omega_h::Write<Omega_h::LO> fieldWrite(vtxRankIdView);
    int width = 2;
    syncStartTime = timer.seconds();
    auto syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), width);
    Kokkos::fence();
    lib.world()->barrier();
    syncTimes.push_back(timer.seconds() - syncStartTime);
 
    deserializeStartTime = timer.seconds();
	  vtxRankId.deserialize(syncFieldRead.view());
    Kokkos::fence();
    deserializeTimes.push_back(timer.seconds() - deserializeStartTime);
  }

/*
  double syncAllSum, syncAllMin, syncAllMax;
  double syncMin = *std::min_element(syncTimes.begin(), syncTimes.end());
  double syncMax = *std::max_element(syncTimes.begin(), syncTimes.end());
  MPI_Allreduce(&syncSum, &syncAllSum, 1, MPI_DOUBLE, MPI_SUM, lib.world()->get_impl()); 
  MPI_Allreduce(&syncMin, &syncAllMin, 1, MPI_DOUBLE, MPI_MIN, lib.world()->get_impl()); 
  MPI_Allreduce(&syncMax, &syncAllMax, 1, MPI_DOUBLE, MPI_MAX, lib.world()->get_impl()); 

  double syncAvg = syncAllSum / numRuns / lib.world()->size();
  double serializeAvg = serializeAllSum / numRuns / lib.world()->size();
  double deserializeAvg = deserializeAllSum / numRuns / lib.world()->size();
*/  

  double syncAllMin, syncAllMax, syncAvg;
  double serializeAllMin, serializeAllMax, serializeAvg;
  double deserializeAllMin, deserializeAllMax, deserializeAvg;
  MPI_Comm comm = lib.world()->get_impl();
  getMinMaxAvg(comm, lib.world()->size(), syncTimes, syncAvg, syncAllMin, syncAllMax);
  getMinMaxAvg(comm, lib.world()->size(), serializeTimes, serializeAvg, serializeAllMin, serializeAllMax);
  getMinMaxAvg(comm, lib.world()->size(), deserializeTimes, deserializeAvg, deserializeAllMin, deserializeAllMax);

  if(lib.world()->rank() == 0) {
  std::cout << "computation, " << "sync, " << "serialize, " << "deserialize" << std::endl <<
               "min, " << syncAllMin << ", " << serializeAllMin << ", " << deserializeAllMin << std::endl <<
               "max, " << syncAllMax << ", " << serializeAllMax << ", " << deserializeAllMax << std::endl <<
               "avg, " << syncAvg << ", " << serializeAvg << ", " << deserializeAvg << std::endl;
  }
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
