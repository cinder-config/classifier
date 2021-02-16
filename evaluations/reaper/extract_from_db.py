import csv

import mysql.connector

REAPER_THRESHOLD = 65

mydb = mysql.connector.connect(
    host="localhost",
    port=8889,
    user="root",
    password="secret_password",
    database="reaper2"
)

cursor = mydb.cursor()
cursor.execute("SELECT project_id,score FROM reaper_results_big")
result = cursor.fetchall()

with open('reaper_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['project', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for x in result:
        writer.writerow({'project': x[0], 'score': "1" if x[1] >= REAPER_THRESHOLD else "0"})