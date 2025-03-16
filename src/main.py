from AnimalNetwork import AnimalNetwork as AN

def main():
    animalNetwork = AN.AnimalNetwork()
    animalNetwork.createNetwork()
    animalNetwork.classifyTestDeeplake()
    animalNetwork.uploadImage()
if __name__ == '__main__':
    main()
