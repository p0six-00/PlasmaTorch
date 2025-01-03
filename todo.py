#TODO: 
#* gestire collisioni con muri 
# normalizzare tutte le simulazioni

#! ci sono dei comportamenti anomali nei pressi delle superfici, fare muri differenziati
#! perchè va così lento con GPU?
#! fixare place uniform

#* implementare collisioni tra particelle 


# calcolo temperatura, calcolo pressione, calcolo correnti


#? iniettori di particelle
#? sonde specifiche localizzate
#? campi/potenziali sovrapposti o aggiunti (cioè che si sommano)
#? muri che diminuiscono potenziale


#? Ristudiare tutta l'elettrostatica
#* aggiungere condizioni di Courant ed altre per avvisare se la simulazione è instabile o menoù
#* fare casi di test per vedere se sto simulando bene:
    #* distribuzione di Maxwell-Boltzmann all'equilibrio
    #* frequenza di plasma
    #* Gridded Ion Thruster 
    #* Condensatore
