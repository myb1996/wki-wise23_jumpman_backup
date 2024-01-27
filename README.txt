*****************************************************
*****************************************************
@author:  Team Jumpman(Yubao Ma, Huanzheng Zhu)

Wettbewerb künstliche Intelligenz in der Medizin

Wintersemester 23/24

KISMED FB18 ETIT TU Darmstadt
*****************************************************
*****************************************************
Im Datei 'CSP' hat Funktion 'CSPMF', es wird zur Berechnung der Projektionsmatrix 'W' verwendet.

Models sind in Datei 'model' gespeichert, bitte die Namen und Path sich nicht ändern.

Im predict_pretrained.py habe ich Input umgeschrieben, bitte sich nicht ändern, test path kann man in Zeile 19 erstellen:

    parser.add_argument('--test_dir', action='store',type=str,default='/shared_data/training_mini')

Bitte predict.py einfach bleiben, nicht sich ändern.

Könnten Sie in score.py Zeile 211:

	parser.add_argument('--test_dir', action='store',type=str,default='../WKIM/shared_data/training_mini/')

die entsprechende REFERENCE.csv von test Sample erstellen.



wettbewerb haben Sie schon gehabt. Bitte hier hinzufügen.

Die benötigten Bibliotheken und Module sowie deren Versionen habe ich bereits in requiements.txt geschrieben. Falls während der Installation Probleme auftritten, installieren Sie sie bitte entsprechend der Version manuell.

Wenn Sie Fragen haben, kontaktieren Sie mich bitte rechtzeitig.

E-mail: yubao.ma@stud.tu-darmstadt.de

Team Jumpman