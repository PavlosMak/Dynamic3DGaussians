import pymeshfix

# Read mesh from infile and output cleaned mesh to outfile
infile = "/media/pavlos/One Touch/datasets/gt_generation/fearless-microwave/generated_mesh.obj"
outfile = "/media/pavlos/One Touch/datasets/gt_generation/fearless-microwave/generated_mesh_fixed.obj"

pymeshfix.clean_from_file(infile, outfile)