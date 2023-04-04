#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <MeshField.hpp>
#include <KokkosController.hpp>

#include <cstdlib>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  OMEGA_H_CHECK(argc == 3);
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(argv[1], lib.world(), &mesh);
  auto dim = mesh.dim();

  using Ctrlr = Controller::KokkosController<MemorySpace, ExecutionSpace, double*>;
  Ctrlr c({mesh.nverts()});
  MeshField::MeshField<Ctrlr> mf(c);

  auto vtxDbls = mf.makeField<0>();

  auto setVtx = KOKKOS_LAMBDA (const int i) {
    vtxDbls(i) = 42.1;
  };

  mf.parallel_for({0,0},{10,10}, setVtx, "set_vertex");

  Omega_h::vtk::write_parallel(argv[2], &mesh, dim);
}
