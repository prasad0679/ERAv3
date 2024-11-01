import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use Agg backend

def visualize_3d_mesh(vertices, faces, title="3D Mesh Visualization"):
    """Create a visualization of the 3D mesh"""
    fig = plt.figure(figsize=(8, 8))  # Adjusted figure size
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh surface
    surf = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces,
                          cmap='viridis',
                          alpha=0.4)
    
    # Plot vertices as red points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              color='red', 
              s=50,
              alpha=1.0,
              label='Vertices')
    
    # Plot edges
    for face in faces:
        points = vertices[face]
        for i in range(3):
            p1 = points[i]
            p2 = points[(i + 1) % 3]
            ax.plot([p1[0], p2[0]], 
                   [p1[1], p2[1]], 
                   [p1[2], p2[2]], 
                   color='#0066FF',
                   linewidth=4.0,
                   alpha=1.0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(['Surface', 'Vertices', 'Edges'])
    ax.view_init(elev=30, azim=45)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{plot_data}' 