import novelpy
import tqdm

# all the cooc possible not including the one done above

ref_cooc = novelpy.utils.cooc_utils.create_cooc(
                 collection_name = "Ref_Journals_sample",
                 year_var="year",
                 var = "c04_referencelist",
                 sub_var = "item",
                 time_window = range(1995,2016),
                 weighted_network = False, self_loop = False)

ref_cooc.main()

ref_cooc = novelpy.utils.cooc_utils.create_cooc(
                 collection_name = "Meshterms_sample",
                 year_var="year",
                 var = "Mesh_year_category",
                 sub_var = "descUI",
                 time_window = range(1995,2016),
                 weighted_network = True, self_loop = True)

ref_cooc.main()

ref_cooc = novelpy.utils.cooc_utils.create_cooc(
                 collection_name = "Meshterms_sample",
                 year_var="year",
                 var = "Mesh_year_category",
                 sub_var = "descUI",
                 time_window = range(1995,2016),
                 weighted_network = False, self_loop = False)

ref_cooc.main()


# Uzzi et al.(2013) Meshterms_sample
for focal_year in tqdm.tqdm(range(2000,2011), desc = "Computing indicator for window of time"):
    Uzzi = novelpy.indicators.Uzzi2013(collection_name = "Meshterms_sample",
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "Mesh_year_category",
                                           sub_variable = "descUI",
                                           focal_year = focal_year,
                                           density = True)
    Uzzi.get_indicator()

# Uzzi et al.(2013) Ref_Journals_sample
for focal_year in tqdm.tqdm(range(2000,2011), desc = "Computing indicator for window of time"):
    Uzzi = novelpy.indicators.Uzzi2013(collection_name = "Ref_Journals_sample",
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "c04_referencelist",
                                           sub_variable = "item",
                                           focal_year = focal_year,
                                           density = True)
    Uzzi.get_indicator()

# Foster et al.(2015) Meshterms_sample
for focal_year in tqdm.tqdm(range(2000,2011), desc = "Computing indicator for window of time"):
    Foster = novelpy.indicators.Foster2015(collection_name = "Meshterms_sample",
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "Mesh_year_category",
                                           sub_variable = "descUI",
                                           focal_year = focal_year,
                                           starting_year = 1995,
                                           community_algorithm = "Louvain",
                                           density = True)
    Foster.get_indicator()

# Lee et al.(2015) Meshterms_sample
for focal_year in tqdm.tqdm(range(2000,2011), desc = "Computing indicator for window of time"):
    Lee = novelpy.indicators.Lee2015(collection_name = "Meshterms_sample",
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "Mesh_year_category",
                                           sub_variable = "descUI",
                                           focal_year = focal_year,
                                           density = True)
    Lee.get_indicator()

# Lee et al.(2015) Ref_Journals_sample
for focal_year in tqdm.tqdm(range(2000,2011), desc = "Computing indicator for window of time"):
    Lee = novelpy.indicators.Lee2015(collection_name = "Ref_Journals_sample",
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "c04_referencelist",
                                           sub_variable = "item",
                                           focal_year = focal_year,
                                           density = True)
    Lee.get_indicator()

# Wang et al.(2017) Meshterms_sample
for focal_year in tqdm.tqdm(range(2000,2011)):
    Wang = novelpy.indicators.Wang2017(collection_name = "Meshterms_sample",
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "Mesh_year_category",
                                           sub_variable = "descUI",
                                           focal_year = focal_year,
                                           time_window_cooc = 3,
                                           n_reutilisation = 1,
                                           starting_year = 1995,
                                           density = True)
    Wang.get_indicator()


# Wang et al.(2017) Ref_Journals_sample
for focal_year in tqdm.tqdm(range(2000,2011)):
    Wang = novelpy.indicators.Wang2017(collection_name = "Ref_Journals_sample",
                                       id_variable = 'PMID',
                                       year_variable = 'year',
                                       variable = "c04_referencelist",
                                       sub_variable = "item",
                                       focal_year = focal_year,
                                       time_window_cooc = 3,
                                       n_reutilisation = 1,
                                       starting_year = 1995,
                                       density = True)
    Wang.get_indicator()
    
