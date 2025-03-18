import spacy

# Diccionario para traducir las etiquetas
etiquetas_dict = {
    'PER': 'Persona',
    'LOC': 'Lugar',
    'ORG': 'Organización',
    'DATE': 'Fecha',
    'TIME': 'Hora',
    'MONEY': 'Dinero',
    'GPE': 'Entidad política',
    'PRODUCT': 'Producto',
    'EVENT': 'Evento',
    'WORK_OF_ART': 'Obra de arte',
    'LANGUAGE': 'Idioma',
    'NORP': 'Grupo político, religión o nacionalidad',
    'FAC': 'Instalación, edificio, etc.',
    'PERCENT': 'Porcentaje',
    'QUANTITY': 'Cantidad',
    'ORDINAL': 'Ordinal',
    'CARDINAL': 'Cardinal',
    'VERB': 'Verbo',
    'ADJ': 'Adjetivo',  # Agregado para identificar los adjetivos
    'NOUN': 'Sustantivo',  # Agregado para identificar los sustantivos
    'ADV': 'Adverbio',  # Agregado para identificar los adverbios
    'PRON': 'Pronombre',  # Agregado para identificar los pronombres
    'DET': 'Determinante',  # Agregado para identificar los determinantes
    'AUX': 'Verbo auxiliar',  # Agregado para identificar los verbos auxiliares
}


# Cargar el modelo en español
nlp = spacy.load("es_core_news_md")

# Texto de prueba
texto = "Adrian Pérez juega al fútbol con Lionel Messi en el estadio Santiago Bernabéu de Madrid."

# Procesar el texto
doc = nlp(texto)

# Mostrar entidades reconocidas con la traducción de la etiqueta
print("\n🔹 Entidades reconocidas:")
for ent in doc.ents:
    label_meaning = etiquetas_dict.get(ent.label_, "Etiqueta desconocida")
    print(f"{ent.text} -> {ent.label_} ({label_meaning})")
    print(f"{ent.text} -> {ent.label_}") 

# Texto adicional para verificar
texto2 = "Adrian juega pelota con el niño por las tardes."
doc2 = nlp(texto2)

# Mostrar entidades reconocidas para el segundo texto
print("\n🔹 Entidades del segundo texto:")
for ent in doc2.ents:
    label_meaning = etiquetas_dict.get(ent.label_, "Etiqueta desconocida")
    print(f"{ent.text} -> {ent.label_} ({label_meaning})")
    print(f"{ent.text} -> {ent.label_}") 
