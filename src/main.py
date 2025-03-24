from AnimalNetwork import AnimalNetwork as AN
from PlantNetwork import PlantNetwork as PN

def main():
    animalNetwork = AN.AnimalNetwork()
    plantNetwork = PN.PlantNetwork()

    # animalNetwork.loadNetwork()
    # #animalNetwork.classifyTestDeeplake()
    # animal = animalNetwork.uploadImage()
    # print(animal)
    # animalNetwork.animalDetails(animal)

    plant = plantNetwork.loadNetwork()
    print(plant)
    plantNetwork.plantDetails(plant)
if __name__ == '__main__':
    main()
