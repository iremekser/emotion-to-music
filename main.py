import emotion_recognizer as tm
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java


ZEMBEREK_PATH = r'data/zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()

print(tm.predict("yaşamayı çok seviyorum bugün çok eğlendim hayatımın en güzel günüydü",morphology))
