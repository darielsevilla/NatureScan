from AnimalNetwork import AnimalNetwork as AN
from PlantNetwork import PlantNetwork as PN
import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Menú")
    root.geometry("300x250")

    def Animal_Scan():
        animalNetwork = AN.AnimalNetwork()
        animalNetwork.loadNetwork()
        #animalNetwork.classifyTestDeeplake()
        animalNetwork.uploadImage()

    def Plant_Scan():
        plantNetwork = PN.PlantNetwork()
        plant = plantNetwork.loadNetwork()
        plantNetwork.plantDetails(plant)

    label = tk.Label(root, text="", font=("Helvetica", 12))
    label.pack(pady=10)
    label = tk.Label(root, text="Select an option:", font=("Helvetica", 12))
    label.pack(pady=10)

    btn1 = tk.Button(root, text="Animal Scan", font=("Helvetica", 10), command=Animal_Scan)
    btn2 = tk.Button(root, text="Plant Scan", font=("Helvetica", 10), command=Plant_Scan)

    btn1.pack(pady=10)
    btn2.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()