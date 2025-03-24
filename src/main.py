from AnimalNetwork import AnimalNetwork as AN
import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Menú")
    root.geometry("300x200")

    def Animal_Scan():
        animalNetwork = AN.AnimalNetwork()
        animalNetwork.loadNetwork()
        #animalNetwork.classifyTestDeeplake()
        animalNetwork.uploadImage()

    def accion_boton2():
        print("Botón 2 presionado")

    label = tk.Label(root, text="Seleccione una opción:", font=("Arial", 12))
    label.pack(pady=10)

    btn1 = tk.Button(root, text="Animal Scan", command=Animal_Scan)
    btn2 = tk.Button(root, text="Botón 2", command=accion_boton2)

    btn1.pack(pady=10)
    btn2.pack(pady=10)

    root.mainloop()
    
if __name__ == '__main__':
    main()
