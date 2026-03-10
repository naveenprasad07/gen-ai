import sqlite3

# connect to sqlite
connection = sqlite3.connect("student.db")

#create a cursor object to insert record, create table
cursor = connection.cursor()

# create the table
table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

# Insert more records 
cursor.execute('''Insert Into STUDENT values('Krish','Data Science','A',90)''')
cursor.execute("INSERT INTO STUDENT VALUES ('Aarav', 'Machine Learning', 'A', 95)")
cursor.execute("INSERT INTO STUDENT VALUES ('Bhavna', 'Data Analytics', 'B', 82)")
cursor.execute("INSERT INTO STUDENT VALUES ('Ishaan', 'Artificial Intelligence', 'A', 88)")
cursor.execute("INSERT INTO STUDENT VALUES ('Sana', 'Cyber Security', 'C', 75)")


## Display all the records 
print("The inserted records are") 
data = cursor.execute('''Select * from STUDENT ''')
for row in data:
    print(row)
    

## Commit your changes in the database
connection.commit()
connection.close()