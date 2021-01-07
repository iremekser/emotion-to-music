import emotion_recognizer as tm

neg, pos = tm.test()
print(neg, pos)

print("Negatif:", neg.count(0) / len(neg))
print("Pozitif:", pos.count(1) / len(pos))
