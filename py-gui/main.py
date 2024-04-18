import os
import pandas as pd
import PySimpleGUI as sg
import datasets.datasets as ds
import user_interaction.main_menu_funcs as mmf
import nl_process.nltk_toolkit as nt
import api_funcs as af


sg.theme('DarkAmber')   # Add a touch of color

# Main window layout
layout = [[sg.Menu([['File', ['Open Single Dataset', 'Open Combined Dataset', 'Configure API settings', 'Search X (Twitter)', 'Load Saved Sentiment Analysis', 'Save', 'Exit']],
                    ['Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['Model Selection', ['nltk']],
                    ['Download', ['Download NLTK Data', 'Download Spacy Data', 'Download Sklearn Data']],
                    ['Recommender', ['Enter Query']],
                    ['View', ['Compare Classifiers', 'Export to PDF', 'S']],
                    ['Help', 'About...'], ])],
                    [sg.Output(size=(150, 40), font='Courier 12', expand_x=True, expand_y=True)],
                    [sg.Button('Run Model', visible=False)],
                    [sg.Button('Run Benchmark Model', visible=False)],
                    [sg.Button('Run Models for Recommender', visible=False)],
                    [sg.Button('Save sentiment results to file', visible=False)],
          [sg.Button('Quit')] ]

# Create the Window

window = sg.Window('MLProject', layout, resizable=True, finalize=True)

# define empty variables to store the dataframe, options and the list of keywords
df = pd.DataFrame()

nltk_options = []

keyword_list = []

column_name_to_tokenize = ''

nltk = None

root = os.path.dirname(os.path.abspath(__file__))

sentiment_df = None

query_keywords = []

recommender_ready = False

tokenized_column = 'cleaned_text_no_stopwords'

selected_columns = ['tweet','favorite_count', 'retweet_count', 'mention_count', 'id', 'mention_total_follower_count', 'quote_retweet_count', 'quote_mention_count']


# Main Event Loop to process events and get the values of the inputs


while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    elif event == 'Quit':
        break
    elif event == 'Open Single Dataset':
        filename = sg.popup_get_file('Dataset to open', no_window=True)
        # read the file into a pandas dataframe
        if filename:
            try:
                df = pd.read_csv(filename)
                column_name_to_tokenize = mmf.datafile_process_text(df)
                keyword_list = mmf.get_keyword_list(df)
                df = mmf.search_df_for_keywords(df, keyword_list)
            except Exception as e:
                sg.popup_error(f'Error reading file: {e}')
    elif event == 'Open Combined Dataset':
            import glob
            csv_file_path = os.path.join(root, 'TwitterArchive', 'TwitterArchive')
            all_dfs = []
            try:
                for file in glob.glob(os.path.join(csv_file_path, '*.csv')):
                    new_df = pd.read_csv(file)
                    all_dfs.append(new_df)
                    print("\nadded file: ", file)
                print("\n\n")
                
                df = pd.concat(all_dfs, ignore_index=True)
                print("\n******* number of rows: *******\n")
                print(len(df))
                print("\n\n")
                df, column_name_to_tokenize = mmf.datafile_process_text(df)
                
                df = df[selected_columns]
                column_name_to_tokenize = 'tweet'
                print("\n\n")
                print("\n******* combined df has {} rows: *******\n".format(len(df)))
                print(df)
            except Exception as e:
                sg.popup_error(f'Error reading file: {e}')
    # model selection menu
    elif event == 'nltk':
        try:
            nltk = nt.NLTKToolkit( root, sg )
            print("\n")
            nltk.display_nltk_data()
            nltk_options = mmf.open_nltk_settings_child_window('nltk model selection')
            print(nltk_options)
            nltk.set_nltk_options(nltk_options)
        except Exception as e:
            sg.popup_error(f'Error creating NLTKToolkit: {e}')
       
    elif event == 'sklearn': # TODO
        mmf.sklearn_model_selection(df)
    elif event == 'spacy':
        mmf.spacy_model_selection(df) # TODO
    elif event == 'Configure API settings':
            try:
                af.configure_api_settings(sg)                
            except Exception as e:
                sg.popup_error(f'Error configuring API settings: {e}')
    # twitter api
    elif event == 'Search X (Twitter)': 
        try:
            api = af.authenticate_with_twitter_api()
            if api != None:
                print('Authenticated with Twitter API')
                search_term = sg.popup_get_text('Enter your search terms, for example a word or phrase')
                tweets_df = af.get_top_tweets(api, search_term)
                print(tweets_df)
        except Exception as e:
            sg.popup_error(f'Error searching Twitter: {e}')
    # nltk download
    elif event == 'Download NLTK Data':
        download_datasets = sg.popup_yes_no('Are you sure you want to download the selected datasets? This will open a seperate dialog')
        if download_datasets == 'Yes':
            try:
                mmf.nltk_download_datasets()
            except Exception as e:
                sg.popup_error(f'Error downloading datasets: {e}')
    elif event == 'Save':
        filename = sg.popup_get_file('Dataset to save', no_window=True)
        if filename:
            try:
                df.to_csv(filename)
            except Exception as e:
                sg.popup_error(f'Error saving file: {e}')
    # Run Model
    elif event == 'Run Model':
                nltk.set_dataframe(df)
                nltk.remove_stopwords_df()
                # run the model with the selected options
                print('running model')
                try:
                    result, column = nltk.run_model()
                    print('** finished **')
                    print(result[ column ])
                except Exception as e:
                    sg.popup_error(f'Error running model: {e}')
    elif event == 'Run Models for Recommender': 
                nltk.set_dataframe(df, column_name_to_tokenize)                             
                try:
                    print('cleaning df')
                    nltk.clean_df()
                    print('cleaned df')
                    nltk.remove_stopwords_df()
                    nltk.lemmatize_words_df()
                    # run the model with the selected options
                    print('running model')
                    sentiment_df, tokenized_column = nltk.run_model()
                    print('\n\n** finished **\n\n')
                    print(sentiment_df.head())
                    print(sentiment_df.tail())
                    # print the number of rows in the result
                    print("\n\nlength of result: ", len(sentiment_df))
                    print("\n=====================================================\n")
                    recommender_ready = True
                    window['Save sentiment results to file'].update(visible=True)
                except Exception as e:
                    sg.popup_error(f'Error running model: {e}')
                    import traceback
                    traceback.print_exc()
    elif event == 'Run Benchmark Model':
            try:
                print(df)
                print('running simple sentiment analysis')
                nltk.set_dataframe(df)
                print(nltk.clean_df())
                df, column = nltk.remove_stopwords_df()
                print(df[ column ])
                sentiment_df, column = nltk.run_nltk_sentiment_analysis()
                print("** finished **")
                print(sentiment_df[ column ])
                recommender_ready = True
                window['Save sentiment results to file'].update(visible=True)
            except Exception as e:
                sg.popup_error(f'Error: {e}')
    elif event == 'Enter Query':
        try:
            if recommender_ready == False:
                sg.popup('Please run the models first')
            else:
                keywords = mmf.get_user_input_for_recommendation()
                if keywords != []:
                    print('keywords: ', keywords)
                    from recommender.recommender_funcs import match_keywords
                    recommended_df = match_keywords(sentiment_df, keywords, tokenized_column, )
                    print("=====================================================")
                    print("\n\nResults of your query: the recommendations are: \n:")
                    print(recommended_df[['tweet', 'id']])
                    #print(recommended_df[['tweet', 'id']])
                    print("=====================================================")
                    # save the df
                    recommended_df.to_csv(os.path.join(root, 'result.csv'))
                    sg.popup('Results of your query have been saved to result.csv')

        except Exception as e:
            sg.popup_error(f'Error: {e}')
            import traceback
            traceback.print_exc()
    elif  event == 'Load Saved Sentiment Analysis':
        filename = sg.popup_get_file('Dataset to open', no_window=True)
        if filename:
            try:
                sentiment_df = pd.read_csv(filename)
                print(sentiment_df)
                recommender_ready = True
            except Exception as e:
                sg.popup_error(f'Error reading file: {e}')
    elif event == 'Save sentiment results to file':
            try:
                filename = sg.popup_get_text('Filename to save as')
                # save the file
                if filename:
                    if not os.path.exists(os.path.join(root, 'saved_sentiment_analysis')):
                        os.makedirs(os.path.join(root, 'saved_sentiment_analysis'))
                    path = os.path.join(root, 'saved_sentiment_analysis', filename) 
                    sentiment_df.to_csv(path)
            except Exception as e:
                sg.popup_error(f'Error saving file: {e}')
    elif event == 'Compare Classifiers':
        try:
            mmf.compare_classifiers()
        except Exception as e:
            sg.popup_error(f'Error comparing classifiers: {e}')
            import traceback
            traceback.print_exc()
    elif event == 'About...':
        sg.popup('About this program', 'Version 0.1', 'Created by: Abdullah Sarwar')
    if not df.empty and nltk_options != []:
        window['Run Model'].update(visible=True)
        window['Run Benchmark Model'].update(visible=True)
        window['Run Models for Recommender'].update(visible=True)
    else:
        window['Run Model'].update(visible=False)
        window['Run Benchmark Model'].update(visible=False)
        window['Run Models for Recommender'].update(visible=False)   
window.close()