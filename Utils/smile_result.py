class SmileResult(object):
    # Class Variable

    # The init method or constructor
    def __init__(self, max_smile, percentage_smile, max_time_of_smile, num_face_detected):
        # Instance Variable
        self.max_smile = max_smile
        self.percentage_smile = percentage_smile
        self.max_time_of_smile = max_time_of_smile
        self.num_face_detected = num_face_detected

    def print_smile_details(self):
        print('% ' + str(self.percentage_smile))
        print('longest smile ' + str(self.max_time_of_smile))
        print('max class ' + str(self.max_smile))

    def get_percentage(self):
        return str(self.percentage_smile) + str(' %')

    def get_max_smile(self):
        return str((self.max_smile*100)) + str(' %')

    def get_max_time_of_smile(self):
        return str(self.max_time_of_smile) + str(' sec')

    def get_num_face_detected(self):
        return self.num_face_detected
