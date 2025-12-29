class DataChecker:
    @staticmethod
    def print_summary(urls, labels):
        print("Number of URLs:\n", len(urls))
        print("Number of labels:\n", len(labels))
        print("Sample URLs:\n", urls[:3])
        print("Sample labels:\n", labels[:3])