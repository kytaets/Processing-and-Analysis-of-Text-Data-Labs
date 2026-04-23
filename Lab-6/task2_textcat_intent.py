"""
Лабораторна робота №6 — Завдання 2
Тема: Розумний дім (Smart Home)
Мета: Навчити компонент TextCategorizer (textcat_multilabel) визначати
       наміри (intent) користувача у системі розумного дому.

Наміри (intents):
  - turn_on      : увімкнути пристрій
  - turn_off     : вимкнути пристрій
  - check_status : перевірити стан пристрою
  - set_value    : встановити значення (яскравість, температура тощо)
  - lock_unlock  : заблокувати / розблокувати
"""

import random
import spacy
from spacy.training import Example
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL

INTENTS = ["turn_on", "turn_off", "check_status", "set_value", "lock_unlock"]

def make_cats(intent: str) -> dict:
    return {i: (1 if i == intent else 0) for i in INTENTS}

TRAIN_DATA = [
    # turn_on
    ("Turn on the lights in the living room.", make_cats("turn_on")),
    ("Switch on the air conditioner.", make_cats("turn_on")),
    ("Please activate the smart speaker.", make_cats("turn_on")),
    ("Start the robot vacuum.", make_cats("turn_on")),
    ("Turn on the thermostat.", make_cats("turn_on")),
    ("Enable the security camera.", make_cats("turn_on")),
    ("Power on the TV.", make_cats("turn_on")),
    ("Can you turn on the ceiling fan?", make_cats("turn_on")),

    # turn_off
    ("Turn off the lights.", make_cats("turn_off")),
    ("Switch off the air conditioner.", make_cats("turn_off")),
    ("Deactivate the motion sensor.", make_cats("turn_off")),
    ("Stop the robot vacuum.", make_cats("turn_off")),
    ("Shut down the smart TV.", make_cats("turn_off")),
    ("Please turn off the smart speaker.", make_cats("turn_off")),
    ("Power off the Wi-Fi router.", make_cats("turn_off")),
    ("Disable the garage door sensor.", make_cats("turn_off")),

    # check_status
    ("Is the front door locked?", make_cats("check_status")),
    ("What is the current temperature?", make_cats("check_status")),
    ("Check the battery level of the smoke detector.", make_cats("check_status")),
    ("Is the security camera working?", make_cats("check_status")),
    ("Show me the status of all devices.", make_cats("check_status")),
    ("Is the garage door open or closed?", make_cats("check_status")),
    ("What is the humidity level right now?", make_cats("check_status")),
    ("Are the lights on in the bedroom?", make_cats("check_status")),

    # set_value
    ("Set the thermostat to 22 degrees.", make_cats("set_value")),
    ("Dim the lights to 50 percent.", make_cats("set_value")),
    ("Increase the brightness of the smart bulb.", make_cats("set_value")),
    ("Set the fan speed to maximum.", make_cats("set_value")),
    ("Lower the temperature by 2 degrees.", make_cats("set_value")),
    ("Adjust the air conditioner to cooling mode.", make_cats("set_value")),
    ("Set the alarm to 7 AM.", make_cats("set_value")),
    ("Change the color of the smart bulb to blue.", make_cats("set_value")),

    # lock_unlock
    ("Lock the front door.", make_cats("lock_unlock")),
    ("Unlock the smart lock.", make_cats("lock_unlock")),
    ("Please lock all the doors.", make_cats("lock_unlock")),
    ("Unlock the garage door for the delivery.", make_cats("lock_unlock")),
    ("Secure the back door.", make_cats("lock_unlock")),
    ("Open the smart lock for my guest.", make_cats("lock_unlock")),
    ("Lock the safe.", make_cats("lock_unlock")),
    ("Can you unlock the front door remotely?", make_cats("lock_unlock")),
]

print("=" * 60)
print("Лабораторна робота №6 — Завдання 2: TextCategorizer (Intent)")
print("=" * 60)
print(f"\nКількість навчальних прикладів : {len(TRAIN_DATA)}")
print(f"Наміри (intents)               : {INTENTS}")

print("\nСтворення моделі spaCy з компонентом textcat_multilabel ...")
nlp = spacy.blank("en")

config = {
    "threshold": 0.5,
    "model": DEFAULT_MULTI_TEXTCAT_MODEL,
}
textcat = nlp.add_pipe("textcat_multilabel", config=config)

for intent in INTENTS:
    textcat.add_label(intent)

train_examples = [
    Example.from_dict(nlp.make_doc(text), {"cats": cats})
    for text, cats in TRAIN_DATA
]

textcat.initialize(lambda: train_examples, nlp=nlp)
print("Компонент TextCategorizer ініціалізовано.")

EPOCHS = 30
print(f"\nПочинаємо навчання ({EPOCHS} епох) ...")

with nlp.select_pipes(enable="textcat_multilabel"):
    optimizer = nlp.resume_training()
    for epoch in range(EPOCHS):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, cats in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"cats": cats})
            nlp.update([example], sgd=optimizer, losses=losses)
        if (epoch + 1) % 5 == 0:
            loss_val = losses.get("textcat_multilabel", 0)
            print(f"  Епоха {epoch + 1:3d}/{EPOCHS}  |  Loss: {loss_val:.4f}")

print("\nНавчання завершено!")

def predict_intent(text: str, threshold: float = 0.4) -> str:
    """Повертає намір з найвищою оцінкою (вище порогу)."""
    doc = nlp(text)
    cats = doc.cats
    best_intent = max(cats, key=cats.get)
    best_score = cats[best_intent]
    if best_score >= threshold:
        return best_intent, best_score
    return "unknown", best_score

TEST_CASES = [
    # (текст, очікуваний намір)
    ("Turn on the bedroom lights.", "turn_on"),
    ("Switch off the smart speaker.", "turn_off"),
    ("What is the thermostat temperature?", "check_status"),
    ("Set the brightness to 80 percent.", "set_value"),
    ("Lock the front door now.", "lock_unlock"),
    ("Activate the security camera.", "turn_on"),
    ("Is the garage door closed?", "check_status"),
    ("Unlock the back door for my friend.", "lock_unlock"),
    ("Lower the fan speed.", "set_value"),
    ("Please turn off all the lights.", "turn_off"),
]

print("\n" + "=" * 60)
print("Результати тестування:")
print("=" * 60)

correct = 0
for text, expected in TEST_CASES:
    predicted, score = predict_intent(text)
    status = "✔" if predicted == expected else "✘"
    if predicted == expected:
        correct += 1
    print(f"\n  {status} Текст    : {text}")
    print(f"    Очікувано : {expected}")
    print(f"    Визначено : {predicted}  (score={score:.3f})")

accuracy = correct / len(TEST_CASES) * 100
print(f"\n{'=' * 60}")
print(f"Точність на тестових прикладах: {correct}/{len(TEST_CASES)} = {accuracy:.1f}%")

print("\n" + "=" * 60)
print("Детальні оцінки (усі наміри) для трьох прикладів:")
print("=" * 60)

DETAIL_EXAMPLES = [
    "Turn on the thermostat.",
    "Lock the garage door.",
    "What temperature is it set to?",
]

for text in DETAIL_EXAMPLES:
    doc = nlp(text)
    print(f"\n  Речення: {text}")
    sorted_cats = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
    for intent, score in sorted_cats:
        bar = "█" * int(score * 20)
        print(f"    {intent:<15} {score:.3f}  {bar}")

print("\n" + "=" * 60)
print("Завдання 2 виконано успішно.")
print("=" * 60)
