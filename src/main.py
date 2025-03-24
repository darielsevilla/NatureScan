from AnimalNetwork import AnimalNetwork as AN
from PlantNetwork import PlantNetwork as PN
import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Menú")
    root.geometry("300x200")

    def Animal_Scan():
        animalNetwork = AN.AnimalNetwork()
        animalNetwork.loadNetwork()
        #animalNetwork.classifyTestDeeplake()
        animal = animalNetwork.uploadImage()
        animalNetwork.animalDetails(animal)

    def accion_boton2():
        plantNetwork = PN.PlantNetwork()
        plant = plantNetwork.loadNetwork()
        plantNetwork.plantDetails(plant)

    label = tk.Label(root, text="Seleccione una opción:", font=("Arial", 12))
    label.pack(pady=10)

    btn1 = tk.Button(root, text="Animal Scan", command=Animal_Scan)
    btn2 = tk.Button(root, text="Botón 2", command=accion_boton2)

    btn1.pack(pady=10)
    btn2.pack(pady=10)

    root.mainloop()
