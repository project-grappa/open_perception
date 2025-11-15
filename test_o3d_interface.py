import open3d as o3d

def build_ui(window):
    em = window.theme.font_size
    margin = 0.5 * em

    # Create a vertical panel on the left
    panel = o3d.visualization.gui.Vert(0, o3d.visualization.gui.Margins(margin, margin, margin, margin))

    # Load and show a 2D image
    image = o3d.io.read_image("path/to/your_image.jpg")
    img_widget = o3d.visualization.gui.ImageWidget(image)
    panel.add_child(img_widget)

    # Add a few buttons
    btn1 = o3d.visualization.gui.Button("Button 1")
    btn1.set_on_clicked(lambda: print("Button 1 clicked!"))
    panel.add_child(btn1)

    btn2 = o3d.visualization.gui.Button("Button 2")
    btn2.set_on_clicked(lambda: print("Button 2 clicked!"))
    panel.add_child(btn2)

    window.add_child(panel)

    # Create a 3D scene widget on the right
    scene_widget = o3d.visualization.gui.SceneWidget()
    scene_widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Load a point cloud
    pcd = o3d.io.read_point_cloud("path/to/pointcloud.pcd")
    material = o3d.visualization.rendering.MaterialRecord()
    scene_widget.scene.add_geometry("pcd", pcd, material)

    # Set a reasonable camera view
    bounds = pcd.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    window.add_child(scene_widget)

    # Define how to lay out the panel and scene side-by-side
    def on_layout(layout_context):
        rect = window.content_rect
        panel_width = 300
        panel.layout(o3d.visualization.gui.Rect(rect.x, rect.y, panel_width, rect.height))
        scene_widget.layout(o3d.visualization.gui.Rect(rect.x + panel_width, rect.y, 
                                                       rect.width - panel_width, rect.height))

    window.set_on_layout(on_layout)

def main():
    o3d.visualization.gui.Application.instance.initialize()
    window = o3d.visualization.gui.Application.instance.create_window(
        "Open3D GUI Example", 1024, 768
    )
    build_ui(window)
    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    main()
