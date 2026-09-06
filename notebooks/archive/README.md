# Archives des expérimentations

Ce répertoire conserve des expérimentations historiques réalisées avant la remise à niveau du projet.

`01_data_exploration_legacy.ipynb` est une copie inchangée du notebook substantiel retrouvé dans `.ipynb_checkpoints`. Il contient de l'EDA, plusieurs essais de modèles, des recherches Optuna et des résultats MLflow historiques.

Ce notebook n'est pas le pipeline officiel. Ses sorties enregistrées proviennent de plusieurs états d'exécution et certaines ne correspondent pas au découpage de données visible dans ses cellules. Elles ne doivent donc pas être utilisées comme référence de production ni comme preuve reproductible de performance.

Le pipeline maintenu se trouve dans `src/` et son orchestration dans `scripts/`. Les futurs notebooks officiels devront appeler ces modules et pouvoir être exécutés intégralement depuis un noyau propre.
