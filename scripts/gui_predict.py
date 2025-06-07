import tkinter as tk
import tensorflow as tf
from tkinter import Canvas, Button, Label
import numpy as np
from PIL import Image, ImageDraw

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        
        # Load your trained model
        self.model = tf.keras.models.load_model("models/mnist_model.h5")
        
        # Drawing canvas
        self.canvas = Canvas(root, width=280, height=280, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Buttons
        self.clear_btn = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT)
        
        self.predict_btn = Button(root, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side=tk.RIGHT)
        
        # Prediction label
        self.label = Label(root, text="Draw a digit and click 'Predict'", font=("Arial", 16))
        self.label.pack()
        
        # Initialize drawing
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
    
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="white", outline="white")
        self.draw.ellipse([x-10, y-10, x+10, y+10], fill=255)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a digit and click 'Predict'")
    
    def predict_digit(self):
        # Convert canvas to MNIST format
        img = self.image.resize((28, 28))  # Resize
        img_array = np.array(img) / 255.0  # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model
        
        # Predict
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        self.label.config(text=f"Predicted: {digit} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
