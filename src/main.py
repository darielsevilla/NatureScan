from AnimalNetwork import AnimalNetwork as AN

def main():
    animalNetwork = AN.AnimalNetwork()
    animalNetwork.createNetwork()
    animalNetwork.classifyTestDeeplake()
if __name__ == '__main__':
    main()
