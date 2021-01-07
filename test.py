import emotion_recognizer as tm
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java

ZEMBEREK_PATH = r'data/zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()

neg, pos = tm.test(morphology)
print(neg, pos)

print("Negatif:", neg.count(0) / len(neg))
print("Pozitif:", pos.count(1) / len(pos))
