from fastai.basic_data import DataBunch
from fastai.data_block import *
from fastai.text.data import *

from pandas import DataFrame

# ======================================================================================================================


# TextList extends ItemList
tl = TextList.from_df(DataFrame(), "path", cols=1, processor=None)
tl.show_xys()


## Special Type
LMTextList()
LMLabelList()


# ItemLists can contain TextLists, used to combine train and valid and test
il = ItemLists(path="path", train=tl, valid=tl)
il.add_test()

# In case of LM, the ItemLists will be convert to a LMTextList and LMLabelList using
TextList.label_for_lm()
# Others
TextList.label_from_df()

# Build ItemLists then the ItemLists can be converted to a DataBunch
LabelLists.databunch()


ItemList()
ItemList.from_df()
ItemList.processor


# ======================================================================================================================

# "Bind `train_dl`,`valid_dl` and `test_dl` in a data object."
db = DataBunch(train_dl=None, valid_dl=None)


tdb = TextDataBunch()

TextDataBunch.load_data()
TextDataBunch.from_df()

#   TDB creates a ItemLists src of Train and Valid
#   ItemLists were converted to
#   The DataBunch is then created using
LabelLists.databunch()
#

TextDataBunch.from_csv()
TextDataBunch.from_ids()
TextDataBunch.from_tokens()

# Extends TextDataBunch
TextLMDataBunch()


TextClasDataBunch()


# ======================================================================================================================

     
2. TokenizeProcessor

3. NumericalizeProcessor

4. ItemLists