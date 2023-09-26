import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex

# Define paths and load the pre-trained model and index
images_folder = 'clothes'
images = os.listdir(images_folder)
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()])
annoy_index = AnnoyIndex(2048, 'angular')
annoy_index.load('clothes_all_50.ann')


class Query:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Search")

        self.query_label = tk.Label(self.root, text="Query Image:")
        self.query_label.pack()

        self.query_button = tk.Button(
            self.root, text="Select Image", command=self.select_query_image)
        self.query_button.pack()

        self.k_label = tk.Label(self.root, text="Number of Nearest Images:")
        self.k_label.pack()

        self.k_entry = tk.Entry(self.root)
        self.k_entry.pack()

        self.search_button = tk.Button(
            self.root, text="Search", command=self.perform_search)
        self.search_button.pack()

        self.image_grid = tk.Label(self.root)
        self.image_grid.pack()

        self.query_image = None
        self.nearest_images = []
        self.query_image_path = None

    def select_query_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.query_image_path = file_path
            self.query_image = Image.open(file_path)
            self.query_image = self.query_image.resize((100, 100))
            self.query_image = ImageTk.PhotoImage(self.query_image)
            self.query_label.config(text="Query Image: Selected")

    def perform_search(self):

        k = int(self.k_entry.get())
        if not self.query_image_path:
            return

        # Perform your nearest neighbor search here and populate self.nearest_images

        # Convert the query_image back to a PIL Image
        query_image_pil = Image.open(self.query_image_path)
        query_image_pil = query_image_pil.resize((100, 100))
        query_image_pil = query_image_pil.convert('RGB')
        input_tensor = transform(query_image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
            self.nearest_images = annoy_index.get_nns_by_vector(
                output_tensor[0].cpu().numpy(), k)
        print(f"Number of Nearest Images: {len(self.nearest_images)}")
        # Create a grid to display the images
        # Calculate the number of rows and columns based on k
        # Adjust the number of columns (e.g., 4 columns per row)
        num_rows = (k + 1) // 4 + 1
        num_cols = min(k + 1, 4)  # Maximum 4 columns per row

        # Calculate grid dimensions based on rows and columns
        grid_width = num_cols * 100
        grid_height = num_rows * 100
        grid = Image.new('RGB', (grid_width, grid_height))

        # Paste the query image
        grid.paste(query_image_pil, (0, 0, 100, 100))

        # Paste the nearest images
        for i, nearest_image_idx in enumerate(self.nearest_images):
            nearest_image_path = os.path.join(
                images_folder, images[nearest_image_idx])

            nearest_image = Image.open(nearest_image_path)
            nearest_image = nearest_image.resize((100, 100))
            nearest_image = nearest_image.convert(
                'RGB')  # Convert to 'RGB' mode
            row = i // 4 + 1
            col = i % 4

            # Calculate the region to paste the nearest image
            left = (col) * 100
            upper = row * 100
            right = left + 100
            lower = upper + 100
            # Specify the region
            grid.paste(nearest_image, (left, upper, right, lower))

        grid = ImageTk.PhotoImage(grid)
        self.image_grid.config(image=grid)
        self.image_grid.image = grid

    def run(self):
        self.root.mainloop()


# Usage
if __name__ == "__main__":
    query_app = Query()
    query_app.run()
