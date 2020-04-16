class SmileResult(object):
    # Class Variable

    # The init method or constructor
    def __init__(self, max_smile, percentage_smile, max_time_of_smile):
        # Instance Variable
        self.max_smile = max_smile
        self.percentage_smile = percentage_smile
        self.max_time_of_smile = max_time_of_smile

    def print_smile_details(self):
        print('% ' + str(self.percentage_smile))
        print('longest smile ' + str(self.max_time_of_smile))
        print('max class ' + str(self.max_smile))

    def get_percentage(self):
        return self.percentage_smile

    def get_max_smile(self):
        return self.max_smile

    def get_max_time_of_smile(self):
        return self.max_time_of_smile
