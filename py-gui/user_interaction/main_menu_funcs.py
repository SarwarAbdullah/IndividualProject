import os
import pandas as pd
import datasets.datasets as ds
import PySimpleGUI as sg
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

def print_separator():
    print('=====================================================')

def datafile_process_text( df ): 
                print_separator()   
                print(f"The column names are: {', '.join(df.columns)}")

                print(f"There are {len(df)} rows in the DataFrame")

                print("The first five rows are:")
                print(df.head())

                column_name = sg.popup_get_text('Enter the name of the column with the text data', default_text='tweet')                
                print("Cleaning the text data in column", column_name)
                df = ds.clean_and_parse_df(df, column_name)
                print(df['cleaned_text'])
                return df, column_name

def get_keyword_list( df ):
                print_separator()
                keyword_list = sg.popup_get_text('Enter the keywords to search for separated by whitespace', default_text='')
                if keyword_list != []:
                    keyword_list = keyword_list.split(' ')
                    keyword_list = [keyword.strip() for keyword in keyword_list]
                    print(f"The keywords you typed are: {keyword_list}")
                else:
                    print("No keywords entered")
                return keyword_list

def search_df_for_keywords( df, keyword_list ):
                print_separator()
                if keyword_list != []:
                    df['keyword_match'] = False
                    for index, row in df.iterrows():
                        for keyword in keyword_list:
                            if keyword in row['cleaned_text']:
                                df.at[index, 'keyword_match'] = True
                    print(f"{len(df[df['keyword_match'] == True])} rows contain one or more of these keywords") 
                    print(df['cleaned_text'][df['keyword_match'] == True])
                    print(df['cleaned_text'], df['keyword_match'])
                else:
                    df['keyword_match'] = True
                return df

def nltk_download_datasets( ):
    cwd = os.getcwd()
    if not os.path.isdir(cwd + '/nltk_data'):
        os.mkdir(cwd + '/nltk_data')
    target_dir = cwd + '/nltk_data'
    result = nltk.download(download_dir=target_dir)
    if result:
        print('Download successful')
    else:
        print('Download failed')

def nltk_classifiers():
    classifiers = ['NaiveBayes', 'DecisionTree', 'SVM', 'KNN', 'RandomForest', 'LogisticRegression', 'GradientBoosting']
    return classifiers

def nltk_corpora():
    corpora = ['twitter_samples', 'movie_reviews', 'opinion_lexicon', 'sentiwordnet', 'product_reviews_1', 'conll2000', 'product_reviews_2', 'aclImdb', 'all']
    return corpora
       
                
def open_nltk_settings_child_window( title ):
    child_values = []

    layout = [[sg.Text('Select the pre-processing steps to apply to the text data')],
                                    [sg.Checkbox('Use Tweet Tokenizer', default=True, key='tweet_tokenizer')],
                                    [sg.Checkbox('Remove stopwords', default=True, key='remove_stopwords')],
                                    [sg.Checkbox('Stem words', default=False, key='stem_words')],
                                    [sg.Checkbox('Lemmatize words', default=True, key='lemmatize_words')],
                                    [sg.Text('Select the nltk classifiers and corpora to use')],
                                    [sg.Listbox(values=nltk_classifiers(), size=(30, 6), key='nltk_classifiers', select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE)],
                                    [sg.Listbox(values=nltk_corpora(), size=(30, 6), key='nltk_corpora', select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
                                    [sg.Button('Save'), sg.Button('Cancel')]]
    
    # create a modal dialog window object
    child_window = sg.Window(title, layout, modal=True)

    # handle the event loop for the child window
    while True:
        child_event, child_values = child_window.read()
        if child_event == sg.WIN_CLOSED or child_event == 'Cancel':
            break
        elif child_event == 'Apply':
            child_window.close()
            break
        elif child_event == 'Save':
            print('saving settings') 
            break
        elif child_event == 'Load':
            filename = sg.popup_get_file('Select a file to load', no_window=True)
            if filename:
                print('loading settings')               
                break
            break
        elif child_event == 'Download NLTK datasets':
            download_datasets = sg.popup_yes_no('Are you sure you want to download the selected datasets? A seperate dialog window will open to show the download progress')
            if download_datasets == 'Yes':
                try: 
                    nltk_download_datasets()
                except Exception as e:
                    sg.popup_error(f'Error downloading datasets: {e}') 
    child_window.close()   
    return child_values

def get_user_input_for_recommendation():
    # recommender dialogue window
    keywords = []
    layout = [[sg.Text('Enter your query and press the button to get recommendations')],
              [sg.Text('The more details you provide, the better the recommendations')],
              [sg.Text('For example, include you preferences of a movie or book')],
              [sg.Text('You can also include a brief description of what you are looking for')],
              [sg.Multiline(size=(50, 10), key='text')],
              [sg.Button('Get Recommendations'), sg.Button('Cancel')]]
    window = sg.Window('Enter text', layout, modal=True)
    # handle the event loop for the window
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        elif event == 'Get Recommendations':
            window.close()
            break
    # import rake-nltk and extract keywords from the user input
    from rake_nltk import Rake
    r = Rake()
    r.extract_keywords_from_text(values['text'])
    keywords = r.get_ranked_phrases()
    print(keywords)
    return keywords


def compare_datasets( combined_df ):
    df_differences = combined_df[combined_df.nunique(axis=1) > 1]
    binary_diff = df_differences.nunique(axis=1) > 1
     # Convert the sentiments to integers: 'positive' to 1 and 'negative' to 0
    df_combined_int = combined_df.replace({'positive': 1, 'negative': 0})

    #create a bar chart to show the agreement between the classifiers
    # Calculate the proportions
    agree = (binary_diff == 0).sum() / len(binary_diff)
    disagree = (binary_diff == 1).sum() / len(binary_diff)

    # Create the bar chart
    plt.bar(['Agree', 'Disagree'], [agree, disagree])
    plt.ylabel('Proportion')
    plt.title('Agreement Between Classifiers')
    plt.show()
    # save the heatmap to a file
    plt.savefig('barchart_differences.png')

    # create a linechart to show the sentiment distribution of the classifiers
    # Create a line chart
    sns.lineplot(data=df_combined_int)
    plt.ylabel('Sentiment')
    plt.title('Sentiment Distribution of Classifiers')
    plt.show()
    # save the linechart to a file
    plt.savefig('linechart_sentiment_distribution.png')

    # from the combined_df, calculate the number of rows where all classifiers agree
    all_agree = (binary_diff == 0).sum()
    # and the number of rows where all classifiers disagree
    all_disagree = (binary_diff == 1).sum()

    # calculate the classifier with the most number of differences between the others
    most_different = binary_diff.sum().idxmax()

    # print the results
    print(f'All classifiers agree on {all_agree} rows') 
    print(f'The classifiers disagree on {all_disagree} rows')
    print(f'The classifier with the most differences is {most_different}')


     
     



def compare_classifiers():
    file_list = []
     # add a dialog window that asks the user to select multiple csv files using th FilesBrowse button
    layout = [[sg.Text('Select the csv files of results to compare. Remember that comparing results from different datasets may not be meaningful.')],
              [sg.Button('Browse', key='files'), sg.Text('No files selected', key='file_list_str')],
              [sg.Button('Compare'), sg.Button('Cancel')]]
    # create a modal dialog window object
    window = sg.Window('Select files', layout, modal=True)
    # handle the event loop for the window
    while True:
        # display the window object and get the values of the inputs
        event, values = window.read()
        # if the user closes the window or clicks cancel
        if event == sg.WIN_CLOSED or event == 'Cancel':
            # break out of the event loop
            break
        elif event == 'files':
            # open a dialog to select multiple files
            files = sg.popup_get_file('Select files to compare', no_window=True, multiple_files=True)
            # if the user selects files
            if files:
                # strip the path from the filenames and store them in a list
                filestr = [os.path.basename(file) for file in files]
                # update the text in the file_list element to show the selected files                
                window['file_list_str'].update(';'.join(filestr))
                # store the list of files
                file_list = files
        # if the user clicks the compare button
        elif event == 'Compare':
            # close the window
            window.close()
            # break out of the event loop
            break
    # create an empty list to store the dataframes
    combined_df = pd.DataFrame()
    # loop through the list of files
    for file in file_list:
        # read the csv file into a dataframe
        df = pd.read_csv(file)
        # append the dataframe to the list
        combined_df[os.path.basename(file)] = df['sentiment']

    print(combined_df)

    if not combined_df.empty:
        compare_datasets( combined_df )
    else:
        print('No data to compare')
