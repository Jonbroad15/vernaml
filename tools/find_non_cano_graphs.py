
    for file in os.listdir('.'):
    g = nx.read_gpickle(file)
    non_cano = 0
    print(file)
    for _, _, data in g.edges(data=True):
        if data['label'] not in ['B53', 'CWW']:
            non_cano += 1
    if non_cano > 10:
        print('more than 10 non_canos: ', file)
        break
     