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
  Kokkos::Profiling::pushRegion("testDist-Dist-Init");
  mesh.ask_dist(0);
  Kokkos::Profiling::popRegion();
  //Use the dist to synchronize values across the vertices - the 'owner' of each
  Kokkos::Profiling::pushRegion("meshField-Serialize");
  auto vtxRankIdView = vtxRankId.serialize();
  Omega_h::Write<Omega_h::LO> fieldWrite(vtxRankIdView);
  int width = 2;
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("meshField-sync");
  MPI_Pcontrol(1);
  auto syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), width);
  MPI_Pcontrol(0);
  Kokkos::Profiling::popRegion();
 
  //replace meshField data with synced values
  Kokkos::Profiling::pushRegion("meshField-Deserialize");
	vtxRankId.deserialize(syncFieldRead.view());
  Kokkos::Profiling::popRegion();
  //check the meshfield Data against the Omega_H ownership array
  auto owners = mesh.ask_owners(0).ranks;
  double error = 0;
  Kokkos::parallel_reduce ("Reduction", mesh.nverts(), KOKKOS_LAMBDA (const int i, double& update) {
	  update += abs(owners[i] - vtxRankId(i, 0));
	  update += abs(owners[i] - vtxRankId(i, 1));
  }, error);
  OMEGA_H_CHECK(error == 0);
  std::cout << "Kokkos Timing Data: " << timer.seconds() << std::endl;
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
