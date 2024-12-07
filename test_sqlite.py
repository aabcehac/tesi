import sqlite3
from datetime import date

### Date handler (conversion to integers)

pot_dates = lambda giorno: (giorno-date(1936,6,24)).days # Ciao, nonna.
date_converter = lambda item: None if item == None or item == '' else pot_dates(date(*[int(i) for i in (item[0:4], item[5:7], item[8:10])]))

### Utilities and Loading data

file = "data\\compas-analysis\\compas.db"
con = sqlite3.connect(file)
cur = con.cursor()
gen = lambda col: cur.execute(f"SELECT {col} FROM people WHERE is_recid != -1") # Exit class: sql.Cursor
gen_list = lambda gen: [val for tup in gen for val in tup]
date_int = lambda lst: list(map(date_converter, lst))
table_gen = lambda data: {col : gen_list(gen(col)) for col in data}
lindt = [*cur.execute("PRAGMA table_info('people')")]
columns = [lindt[_][1] for _ in range(len(lindt))]
data = {col : gen_list(gen(col)) for col in columns}

none_number = lambda caller, data, columns: print(*[f"{caller}" + f"{i}: " + f"Nones: {data[i].count(None)} " + f"Uniques: {len(set(data[i]))} " + f"Total: {len(data[i])}" for i in columns], "", sep="\n")
people = ('id', 'first', 'last')
people_data = table_gen(people)

### Proiezione algebrica dei dati

short_columns = ('id', 'sex', 'race', 'dob', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_recid', 'decile_score')
short_data = table_gen(short_columns)
# none_number('projected_data: ', short_data, short_columns) # projected data structure
assert data['is_recid'].count(-1) == 0
#print("\'decile_score\': ", data['decile_score'].count(-1)) # Output: 15

race_uniques = [*set(short_data['race'])]
race_list = [f"is_{i.lower()}" for i in race_uniques]
transformed = ('is_male', 'dob_tr', *race_list)
# data['id'] = [data['id'][i] -1 for i in range(len(data['id']))]
short_data['id'] = [item-1 for item in short_data['id']]
assert (short_data['id'][i] == i for i in range(len(short_data['id'])))
short_data['is_male'] = [int(item == 'Male') for item in short_data['sex']]
short_data['dob_tr'] = [date_converter(item) for item in short_data['dob']]
for i in range(len(race_uniques)):
    short_data[race_list[i]] = [int(item == race_uniques[i]) for item in short_data['race']]
# none_number('transformed_data: ', short_data, transformed) # revised columns
kys = [*short_data.keys()]
#print(kys)
kys.remove('sex')
kys.remove('race')
kys.remove('dob')
kys.append(kys.pop(6))
kys.append(kys.pop(6))
#print(*kys, sep="\n")
ultimate_data_dict = {i : short_data[i] for i in kys}
for key, values in ultimate_data_dict.items():
    for value in values:
        assert type(value) == int
ultimate_data_list = [*ultimate_data_dict.values()]
# print(ultimate_data_list)
