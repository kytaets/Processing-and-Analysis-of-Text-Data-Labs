"""
Лабораторна робота №6 — Завдання 1
Тема: Розумний дім (Smart Home)
Мета: Додати власний тип сутності DEVICE (пристрій розумного дому)
       до існуючої моделі spaCy та навчити модель розпізнавати її.
"""

import json
import random
import spacy
from spacy.training import Example

TRAIN_JSON = [
    {
        "sentence": "Turn on the thermostat in the living room.",
        "entities": [{"value": "thermostat", "label": "DEVICE", "start": 11, "end": 21}]
    },
    {
        "sentence": "The smart lock is unlocked.",
        "entities": [{"value": "smart lock", "label": "DEVICE", "start": 4, "end": 14}]
    },
    {
        "sentence": "Please dim the ceiling light.",
        "entities": [{"value": "ceiling light", "label": "DEVICE", "start": 15, "end": 28}]
    },
    {
        "sentence": "Switch off the air conditioner.",
        "entities": [{"value": "air conditioner", "label": "DEVICE", "start": 15, "end": 30}]
    },
    {
        "sentence": "Set the security camera to night mode.",
        "entities": [{"value": "security camera", "label": "DEVICE", "start": 8, "end": 23}]
    },
    {
        "sentence": "The smoke detector triggered an alarm.",
        "entities": [{"value": "smoke detector", "label": "DEVICE", "start": 4, "end": 18}]
    },
    {
        "sentence": "Activate the robot vacuum in the hallway.",
        "entities": [{"value": "robot vacuum", "label": "DEVICE", "start": 13, "end": 25}]
    },
    {
        "sentence": "Turn off the smart speaker.",
        "entities": [{"value": "smart speaker", "label": "DEVICE", "start": 13, "end": 26}]
    },
    {
        "sentence": "The garage door is still open.",
        "entities": [{"value": "garage door", "label": "DEVICE", "start": 4, "end": 15}]
    },
    {
        "sentence": "Check the doorbell camera feed.",
        "entities": [{"value": "doorbell camera", "label": "DEVICE", "start": 10, "end": 25}]
    },
    {
        "sentence": "Increase the brightness of the smart bulb.",
        "entities": [{"value": "smart bulb", "label": "DEVICE", "start": 31, "end": 41}]
    },
    {
        "sentence": "The water sensor detected a leak.",
        "entities": [{"value": "water sensor", "label": "DEVICE", "start": 4, "end": 16}]
    },
    {
        "sentence": "Mute the smart TV in the bedroom.",
        "entities": [{"value": "smart TV", "label": "DEVICE", "start": 9, "end": 17}]
    },
    {
        "sentence": "The motion sensor is active.",
        "entities": [{"value": "motion sensor", "label": "DEVICE", "start": 4, "end": 17}]
    },
    {
        "sentence": "Reset the Wi-Fi router.",
        "entities": [{"value": "Wi-Fi router", "label": "DEVICE", "start": 10, "end": 22}]
    },
]

print("=" * 60)
print("Лабораторна робота №6 — Завдання 1: NER (DEVICE)")
print("=" * 60)

def json_to_spacy_format(json_data):
    """Перетворює JSON-анотації у формат spaCy."""
    result = []
    for item in json_data:
        text = item["sentence"]
        entities = [
            (ent["start"], ent["end"], ent["label"])
            for ent in item["entities"]
        ]
        result.append((text, {"entities": entities}))
    return result

TRAIN_DATA = json_to_spacy_format(TRAIN_JSON)

print("\nПриклади навчальних даних (перші 3):")
for text, ann in TRAIN_DATA[:3]:
    print(f"  Текст : {text}")
    print(f"  Сутності: {ann['entities']}")
    print()

print("Завантаження моделі en_core_web_md ...")
nlp = spacy.load("en_core_web_md")

ner = nlp.get_pipe("ner")
ner.add_label("DEVICE")

print(f"Наявні мітки NER після додавання: {ner.labels}")

EPOCHS = 30
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

print(f"\nПочинаємо навчання ({EPOCHS} епох) ...")
print(f"Вимкнені компоненти під час навчання: {other_pipes}")

losses_history = []

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()
    for epoch in range(EPOCHS):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotation in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotation)
            nlp.update([example], sgd=optimizer, losses=losses)
        losses_history.append(losses.get("ner", 0))
        if (epoch + 1) % 5 == 0:
            print(f"  Епоха {epoch + 1:3d}/{EPOCHS}  |  NER loss: {losses.get('ner', 0):.4f}")

print("\nНавчання завершено!")

ner.to_disk("new_ner_device")
print("Компонент NER збережено у папку 'new_ner_device'.")

TEST_SENTENCES = [
    "Please turn on the thermostat.",
    "The robot vacuum needs charging.",
    "I want to check the security camera.",
    "The smoke detector is making noise.",
    "Dim the smart bulb in the kitchen.",
    "Is the garage door closed?",
    "Turn off the air conditioner before leaving.",
]

print("\n" + "=" * 60)
print("Результати тестування навченої моделі:")
print("=" * 60)

for sent in TEST_SENTENCES:
    doc = nlp(sent)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    device_entities = [e for e in entities if e[1] == "DEVICE"]
    print(f"\n  Речення : {sent}")
    if device_entities:
        for text, label in device_entities:
            print(f"  ✔ Знайдено [{label}]: «{text}»")
    else:
        print("  — Пристрої не знайдено")

print("\n" + "=" * 60)
print("Завдання 1 виконано успішно.")
print("=" * 60)
