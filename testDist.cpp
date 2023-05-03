#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <MeshField.hpp>
#include <KokkosController.hpp>
#include <span>

#include <cstdlib>

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
  using Ctrlr = Controller::KokkosController<MemorySpace, ExecutionSpace, int*>;
  Ctrlr c({mesh.nverts()});
  MeshField::MeshField<Ctrlr> mf(c);

  auto vtxRankId = mf.makeField<0>();

  const int rank = lib.world()->rank();
  auto setVtx = KOKKOS_LAMBDA (const int i) {
    vtxRankId(i) = rank;
  };
  mf.parallel_for({0},{mesh.nverts()}, setVtx, "set_vertex");

  auto dist = mesh.ask_dist(0);
  //Use the dist to synchronize values across the vertices - the 'owner' of each
  //vertex has its value become the value on the non-owning processes 
  
  auto fieldPtr = vtxRankId.data(); //Has this been added yet?
  auto fieldSize = mesh.nverts(); //Should be vtxRankId.size() -> but fine for this example
  
  Kokkos::View<int*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>
  fieldData(fieldPtr, fieldSize);

  Omega_h::Write fieldWrite(fieldData);
  int width = 1;
  auto syncFieldRead = mesh.sync_array(fieldSize, Omega_h::Read(fieldWrite), width);
 
  //TODO: replace meshField data with synced values

  //convert the meshfield to an omegah 'tag' for visualization
  Omega_h::Write<Omega_h::LO> vtxVals(mesh.nverts());
  auto mfToOmegah = KOKKOS_LAMBDA (const int i) {
    vtxVals[i] = vtxRankId(i);
  };
  mf.parallel_for({0},{mesh.nverts()}, mfToOmegah, "meshField_to_omegah");
  mesh.add_tag(0, "fromMeshFieldInt", 1, Omega_h::read(vtxVals));

  //write vtk files
  Omega_h::vtk::write_parallel(argv[2], &mesh, mesh.dim());
  return 0;
}
