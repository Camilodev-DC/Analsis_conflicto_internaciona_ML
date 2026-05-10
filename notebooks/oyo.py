from acled import AcledClient

# Inicias sesión con tu correo y contraseña del nuevo portal
client = AcledClient(username="brayan.hernandez3@est.uexternado.edu.co", password="q*so%X4Rlk!5*1nNS%LFklK7v")

# Descargas los datos
eventos = client.get_data(
    country='Colombia',
    year=2024,
    limit=100
)

for e in eventos:
    print(e['event_date'], e['event_type'], e['notes'])