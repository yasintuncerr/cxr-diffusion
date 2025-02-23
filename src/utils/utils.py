

def print_dict_as_table(dictionary, titles = ["Key", "Count"]):
    # Print header
    print(f"{titles[0]:<20} | {titles[1]:<10}")
    print("-" * 32)
    
    # Print each key-value pair
    for key, value in dictionary.items():
        print(f"{str(key):<20} | {str(value):<10}")