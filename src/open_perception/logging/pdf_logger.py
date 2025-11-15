import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
import os


class PDFReport(FPDF):
    def __init__(self, title="VLM Report"):
        super().__init__()
        self.set_title(title)
        self.img_count=0
        
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, self.title, 0, 1, "C")
        self.ln(5)
        # pass
    

    def add_text(self, text, color=(0, 0, 0)):
        self.set_text_color(*color)
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 8, text)
        self.ln(1)

    def add_image_from_array(self, image_array):
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(image_array[:,:,::-1])

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        with open(f"temp_image_{self.img_count}.png", "wb") as f:
            f.write(buf.getbuffer())

        self.image(f"temp_image_{self.img_count}.png", h=80)
        self.ln(1)
        os.remove(f"temp_image_{self.img_count}.png")
        self.img_count += 1


if __name__ == "__main__":
    content = [
        "normal text",
        ("some more text in yellow..." * 20, (255, 200, 0)),
        np.random.rand(100, 100, 3),
        ("more text but in blue" * 100, (0, 0, 255)),
        np.random.rand(100, 100, 3),
    ]

    pdf = PDFReport()
    pdf.add_page()

    for item in content:
        if isinstance(item, str):
            pdf.add_text(item)
        elif isinstance(item, tuple) and isinstance(item[0], str):
            pdf.add_text(item[0], item[1])
        elif isinstance(item, np.ndarray):
            pdf.add_image_from_array(item)

    pdf.output("report.pdf")
