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

  const int rank = lib.world()->rank();
  
  Omega_h::Write<Omega_h::LO> fieldWrite(mesh.nverts(), rank);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("meshField-sync");
  MPI_Pcontrol(1);
  auto syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  syncFieldRead = mesh.sync_array(0, Omega_h::Read(fieldWrite), 1);
  MPI_Pcontrol(0);
  Kokkos::Profiling::popRegion();
  
  std::cout << "Kokkos Timing Data: " << timer.seconds() << std::endl;
  return 0;
}
