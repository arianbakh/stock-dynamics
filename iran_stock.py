import json
import numpy as np
import os
import pickle
import requests
import sys

from bs4 import BeautifulSoup

from network import Network
from settings import DATA_DIR


AUTOCOMPLETE_URL = 'http://www.fipiran.com/DataService/AutoCompleteindex'
EXPORT_URL = 'http://www.fipiran.com/DataService/Exportindex'
START_DATE = 13980101  # YYYYMMDD Solar Hijri calendar
END_DATE = 13990101  # YYYYMMDD Solar Hijri calendar


class IranStock:
    def __init__(self):
        self._instrument_id_to_node_index = {}
        self._instrument_ids = []
        self._names = []
        self._date_to_time_frame_index = {}
        self._dates = []
        self._raw_data = None
        self._get_raw_data()
        self._fill_raw_data_empty_entries()
        self._delete_static_columns()
        self.network = None
        self._create_network()

    @staticmethod
    def _get_stock_indices():
        all_instrument_ids = []
        all_names = []
        cached_path = os.path.join(DATA_DIR, 'iran_stock_indices.json')
        if os.path.exists(cached_path):
            with open(cached_path, 'r') as cached_file:
                response_json = json.loads(cached_file.read())
        else:
            response = requests.post(AUTOCOMPLETE_URL, data={
                'id': '',
            })
            if response.status_code == 200:
                with open(cached_path, 'w') as cached_file:
                    cached_file.write(response.text)
                response_json = json.loads(response.text)
            else:
                response_json = []

        for item in response_json:
            name = item['LVal30']
            if name[0].isdigit():
                instrument_id = item['InstrumentID']
                all_instrument_ids.append(instrument_id)
                all_names.append(name)

        return all_instrument_ids, all_names

    def _get_raw_data(self):
        dates = set()
        entries = []
        all_instrument_ids, all_names = self._get_stock_indices()
        node_counter = 0
        for i, instrument_id in enumerate(all_instrument_ids):
            # progress bar
            sys.stdout.write('\rExcel Files [%d/%d]' % (i + 1, len(all_instrument_ids)))
            sys.stdout.flush()

            cached_path = os.path.join(DATA_DIR, instrument_id)
            if os.path.exists(cached_path):
                with open(cached_path, 'r') as cached_file:
                    response_text = cached_file.read()
            else:
                response = requests.post(EXPORT_URL, data={
                    'inscodeindex': instrument_id,
                    'indexStart': START_DATE,
                    'indexEnd': END_DATE,
                })
                response_text = response.text
                with open(cached_path, 'w') as cached_file:
                    cached_file.write(response_text)

            soup = BeautifulSoup(response_text, features='html.parser')
            table = soup.find('table')
            if table:
                for row in table.findAll('tr'):
                    if row:
                        columns = row.findAll('td')
                        if columns:
                            date = columns[1].string.strip()
                            dates.add(date)
                            amount = columns[2].string.strip()
                            entries.append((date, instrument_id, amount))
                self._instrument_ids.append(instrument_id)
                self._instrument_id_to_node_index[instrument_id] = node_counter
                node_counter += 1
                self._names.append(all_names[i])
        print()  # newline

        self._dates = sorted(dates)
        for i, date in enumerate(self._dates):
            self._date_to_time_frame_index[date] = i

        self._raw_data = np.zeros((len(self._dates), len(self._instrument_ids)))
        for entry in entries:
            date = entry[0]
            time_frame_index = self._date_to_time_frame_index[date]
            instrument_id = entry[1]
            node_index = self._instrument_id_to_node_index[instrument_id]
            amount = entry[2]
            self._raw_data[time_frame_index, node_index] = amount

    def _fill_raw_data_empty_entries(self):
        for column_index in range(self._raw_data.shape[1]):
            previous_value = 0
            for row_index in range(self._raw_data.shape[0]):
                if not self._raw_data[row_index, column_index]:
                    self._raw_data[row_index, column_index] = previous_value
                else:
                    previous_value = self._raw_data[row_index, column_index]

    @staticmethod
    def _subtract(x):
        normalized_columns = []
        for column_index in range(x.shape[1]):
            column = x[:, column_index]
            normalized_column = column - column[0]
            normalized_columns.append(normalized_column)
        normalized_x = np.column_stack(normalized_columns)
        return normalized_x

    def _delete_static_columns(self):
        normalized_x = self._subtract(self._raw_data)
        mask = (normalized_x == 0).all(0)
        column_indices = np.where(mask)[0]
        new_names = []
        new_instrument_ids = []
        for i in range(len(self._names)):
            name = self._names[i]
            instrument_id = self._instrument_ids[i]
            if i not in column_indices:
                new_names.append(name)
                new_instrument_ids.append(instrument_id)
        self._raw_data = self._raw_data[:, ~mask]

    def _create_network(self):
        self.network = Network(self._raw_data, self._dates, self._instrument_ids)


def get_iran_stock_network(recreate=False):
    cached_path = os.path.join(DATA_DIR, 'iran_stock_network.p')
    if not recreate and os.path.exists(cached_path):
        with open(cached_path, 'rb') as cached_file:
            network = pickle.load(cached_file)
    else:
        network = IranStock().network
        with open(cached_path, 'wb') as cached_file:
            pickle.dump(network, cached_file)
    return network
