# export_points_ply: saves a ply file containing coloured labelled points
def export_points_ply(filepath, points):
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %d\n'%(points.shape[0]))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar diffuse_red\n')
    f.write('property uchar diffuse_green\n')
    f.write('property uchar diffuse_blue\n')
    f.write('end_header\n')
    for i in range(points.shape[0]):
        if points.shape[1] == 4:
            if points[i,3] == 0:
                (R,G,B) = (0,255,0)
            else:
                (R,G,B) = (255,0,0)
        else:
            (R,G,B) = (255,0,0)
        f.write('%.4f %.4f %.4f %d %d %d\n'%(points[i,0],points[i,1],points[i,2],int(R),int(G),int(B)))
    f.close()
