
"""

Grado en Ingeniería de Sistemas Audiovisuales y Multimedia
Universidad Rey Juan Carlos - Campus Fuenlabrada

TFG Aplicación de Ciencia de Datos en Inversiones Inmobiliarias
Tutor: Rebeca Goya Esteban

Autor: Carlos Gaitán Gil

"""

import time
import matplotlib
import requests
import re
from bs4 import BeautifulSoup
import html5lib
import webbrowser
import imaplib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import html # html.unescape -> Saving Alcorcón instead of Alcorc&#243;n
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler




def getPropertiesHabitaclia():
"""
Function to extract, clean, transform and save data from Habitaclia.com
"""

    url = 'https://www.habitaclia.com/alquiler-en-zona_sur.htm'
    #url = 'https://www.habitaclia.com/viviendas-en-zona_sur.htm'
    headers = ({
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
    })

    properties = []

    # LOOP FOR OTHER PAGES
    no_more_pages = False
    while not no_more_pages:

        try:
            r = requests.get(url, headers = headers)
            source_code = r.text
        except ConnectionError as err:
            print(err)
            quit()

        patt_str = '<a\shref="(.*?)".*?>(.*?)</a>\s+</h3>\s+<p.*?>\s+<span>(.*?)</span>|</p>\s+<p.*?>\s+(.*?)m<sup>2</sup>\s+-\s(.*?)\sh.*?\s-\s(.*?)\sb.*?\s-\s(.*?)€/m<sup>2</sup>|<span.*?itemprop="price">(.*?)\s€</span>|<span.*?price-down">ha bajado (.*?)\s€</span>'
        patt = re.compile(patt_str)
        properties_aux = re.findall(patt, source_code)
        if properties_aux:
            properties += properties_aux

            # Checking if there's another web page to keep taking properties from
            next_patt_str = '<li\sclass="next">\s+<a\shref="(.*?)">\s+<span.*?>\s+Siguiente\s+</span>'
            next_patt = re.compile(next_patt_str)
            next = re.findall(next_patt, source_code)
            if next:
                url = next[0]
            else:
                no_more_pages = True

    # List of duples into list of lists
    properties = [list(property) for property in properties]

    properties_aux = []

    # Putting all the info of a house in the same list inside the main one
    # as well as cleaning empty elements
    for i in range(0, len(properties)-1):
        # For every position whith an url as the first element but the next one has not an url as well
        if properties[i][0][:5] == 'https' and properties[i+1][0][:5] != 'https':
            properties_aux.append(  [element for element in properties[i] if element]+
                                    [element for element in properties[i+1] if element]+
                                    [element for element in properties[i+2] if element])
            if i != len(properties)-3:
                if properties[i+3][0][:5] != 'https':
                    properties_aux[-1] = properties_aux[-1] + [element for element in properties[i+3] if element]

    properties = properties_aux

    fixed_properties = []
    for property in properties:
        if len(property) in [8, 9]:
            fixed_property = []
            # URL
            fixed_property.append(property[0])
            # Type
            if property[1].find('Alquiler') == 0:
                property[1] = property[1][9:]
            fixed_property.append(property[1][:property[1].find('  ')]) # Format: [Type][two spaces]['en  ' or Place]
            if property[1].find('  en  ') != -1:
                property[1] = property[1][property[1].find('  ')+6:]
            else:
                property[1] = property[1][property[1].find('  ')+2:]
            #Specific location plus info
            if property[1].find('.') != -1: # Theres info after titile
                fixed_property.append(property[1][:property[1].find('.')])
                fixed_property.append(property[1][property[1].find('.')+2:])
            else:
                fixed_property.append(property[1])
                fixed_property.append('NaN')
            if property[2].find('-') != -1: # 'Alcorcón - Centro'
                fixed_property.append(html.unescape(property[2][:property[2].find(' ')])) # Localidad
                fixed_property.append(html.unescape(property[2][property[2].find('-')+2:]))
            else: # 'Alcorcón'
                fixed_property.append(html.unescape(property[2]))
                fixed_property.append(html.unescape(property[2]))
            fixed_property.append(property[3]) # m*2
            fixed_property.append(property[4]) # Hab
            fixed_property.append(property[5]) # Baños
            # Precio/m*2
            if (property[6].find(',') == -1) and (property[6].replace('.', '').isnumeric()): # Viviendas en venta: '1.000' -> 1000.0
                fixed_property.append(float(property[6].replace('.', '')))
            elif (property[6].find(',') != -1) and (property[6].replace(',', '').isnumeric()): # Viviendas en alquiler: '10,10' -> 10.10
                fixed_property.append(float(property[6].replace(',', '.')))
            else:
                fixed_property.append(property[6])
            # Precio
            if property[7].replace('.', '').isnumeric():
                fixed_property.append(int(property[7].replace('.', '')))
            else:
                fixed_property.append(property[7])
            # Disccount
            if len(property) == 8:
                fixed_property.append('No')
            elif len(property) == 9:
                if property[8].replace('.', '').isnumeric():
                    fixed_property.append(int(property[8].replace('.', '')))
                else:
                    fixed_property.append(property[8])
            #Elevator, Other info from [1]
            if property[1].find('ascensor') != -1:
                fixed_property.append('Sí')
            else:
                fixed_property.append('-')

            fixed_properties.append(fixed_property)

    index = list(range(1, len(fixed_properties)+1))
    properties_df = pd.DataFrame(fixed_properties, index, columns = ['URL',
                                                         'TIPO',
                                                         'UBICACIÓN',
                                                         'INFO',
                                                         'LOCALIDAD',
                                                         'ZONA',
                                                         'M*2',
                                                         'HAB',
                                                         'BAÑOS',
                                                         'PRECIO/m*2',
                                                         'PRECIO',
                                                         'DESCUENTO',
                                                         'ASCENSOR'])

    properties_df = properties_df[(properties_df['M*2'].apply(lambda x: x.isnumeric())) &
                                  (properties_df['HAB'].apply(lambda x: x.isnumeric())) &
                                  (properties_df['BAÑOS'].apply(lambda x: x.isnumeric()))]

    properties_df['M*2'] = properties_df['M*2'].astype('int')
    properties_df['BAÑOS'] = properties_df['BAÑOS'].astype('int')
    properties_df['HAB'] = properties_df['HAB'].astype('int')

    cleaned_data_with_pred.to_csv('habitaclia.csv', index=False)
    #cleaned_data_with_pred.to_csv('habitaclia_madridsur_compra.csv', index=False)

    cleaned_data_with_pred.to_csv('habitaclia_madridsur_rentabilidad.csv', index=False)

def getPropertiesBS():
"""
Data extraction using BeautifulSoup and Regular Expression
"""

    url = 'https://www.habitaclia.com/alquiler-en-zona_sur.htm'
    headers = ({
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    try:
        r = requests.get(url, headers = headers)
        source_code = r.text
    except ConnectionError as err:
        print(err)
        quit()

    #soup = BeautifulSoup(r.text, features='html5lib')

    # Saving first page's source code in local .txt file
    with open('habitaclia_page.txt', 'w') as file:
        file.write(r.text)

    # Loading source code in local .txt file
    with open('habitaclia_page.txt', 'r') as file:
        page = file.read()

    soup = BeautifulSoup(page, features='html5lib')
    lst = soup.find_all('div', class_='list-item-info')

    first_elem = lst[0]

    print(str(first_elem.h3.a.attrs['href']))  # Same result as with -> [str(lst.h3.a).find('https'):str(lst.h3.a).find('" ')]) # URL
    print(first_elem.h3.a.string) # Title
    print(first_elem.find('p', ['list-item-location']).span.string) # Location
    info = str(first_elem.find('p', ['list-item-feature']))
    patt = '<p.*?>\s+(.*?)m<sup>2</sup>\s+-\s(.*?)\shab.*?\s+-\s(.*?)\sba.*?\s+-\s.*?\s+</p>'
    info_lst = re.findall(patt, info)[0]
    print(info_lst) # square meter + rooms + bathrooms
    print(first_elem.find('article', ['list-item-price']).span.string) # Price
