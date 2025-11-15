import open3d as o3d



import open3d as o3d
import numpy as np


r = 1
height = 2

cone = o3d.geometry.TriangleMesh.create_cone(radius=r, height=height)
cone.translate([2, 0, 0]-cone.get_center())
cone = o3d.t.geometry.TriangleMesh.from_legacy(cone)

shifted_cone = o3d.geometry.TriangleMesh.create_cone(radius=r, height=height) 
shifted_cone.translate(np.array([2,0,-height*0.05]) - shifted_cone.get_center())
shifted_cone = o3d.t.geometry.TriangleMesh.from_legacy(shifted_cone)
cone = cone.boolean_difference(shifted_cone)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
frame = o3d.t.geometry.TriangleMesh.from_legacy(frame)
def display_with_edges(meshes):
    geometries = []
    for i,m in enumerate(meshes):
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(m.to_legacy())
        lines.paint_uniform_color([1, 0, 0])  # Red color for edges
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        mat.base_color = [0, 1, 0, 0.5]  # Green color with 50% transparency

        geometries.extend([{"name":f"mesh_{i}", "geometry": m.to_legacy(), "material": mat},
                                       {"name":"edges", "geometry": lines}])
    return geometries

o3d.visualization.draw(display_with_edges([cone, frame]))


sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
sphere.translate([0, 0, 0]-sphere.get_center())
sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

sphere_2 = o3d.geometry.TriangleMesh.create_sphere(0.7)
sphere_2.translate([0, 0, 0]-sphere_2.get_center())
sphere_2 = o3d.t.geometry.TriangleMesh.from_legacy(sphere_2)

box = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=10)
box.translate([0, 0, 0]-box.get_center())
box = o3d.t.geometry.TriangleMesh.from_legacy(box)

sphere2 = sphere_2.boolean_union(box)
o3d.visualization.draw(display_with_edges([sphere2]))

# sphere = sphere.boolean_difference(sphere_inside).to_legacy()


# intersection = box.boolean_intersection(sphere)
# o3d.visualization.draw(display_with_edges([intersection]))
# intersection = o3d.t.geometry.TriangleMesh.from_legacy(intersection)
# union = box.boolean_union(sphere_2)
diff2 = sphere.boolean_difference(sphere_2)
o3d.visualization.draw(display_with_edges([diff2]))

difference = box.boolean_difference(sphere)
o3d.visualization.draw(display_with_edges([difference]))
union = box.boolean_union(sphere)
o3d.visualization.draw(display_with_edges([union]))