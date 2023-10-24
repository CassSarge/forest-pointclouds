# export_points_ply: saves a ply file containing coloured labelled points
def export_points_ply(filepath, points):
    print("Exporting points to ply file {}...".format(filepath))
    
    f = open(filepath, "w");
    # Write the header
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
    for i in range(points.shape[0]): # For each point
        if points.shape[1] == 4: # If there are labels
            if points[i,3] == 0: # Foliage
                (R,G,B) = (0,255,0) # Green
            elif points[i,3] == 1: # Stem
                (R,G,B) = (255,0,0) # Red
            elif points[i,3] == 2: # Ground
                (R,G,B) = (0,0,255) # Blue
            elif points[i,3] == 3: # Undergrowth
                (R,G,B) = (0,255,255) # Cyan
        f.write('%.4f %.4f %.4f %d %d %d\n'%(points[i,0],points[i,1],points[i,2],int(R),int(G),int(B)))
    f.close()

if __name__ == '__main__':
    import pickle
    import numpy as np
    with open("data/plot_annotations.p", "rb") as f:
        data = pickle.load(f)
        data = np.asarray(data)
    export_points_ply("data/test_plot_annotations.ply", data)