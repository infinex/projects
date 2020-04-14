"""
Configuration for the Database Tables
"""
# SQL QUERY
SQL_QUERY_STRING = "select complaints_users.*,products.main_product,products.sub_product from complaints_users " \
                   "left join products on complaints_users.product_id = products.product_id"
PRODUCT_QUERY_STRING = "select * from products"

# CONFIG NAMING CONVENTION AND FEATURES
# Main Features
PRODUCT_ID = 'product_id'
MAIN_PRODUCT = 'main_product'
SUB_PRODUCT = 'sub_product'
COMPLAINT_TEXT = 'complaint_text'
COMPLAINT_ID = 'complaint_id'

# New Features
F_COMPLAINT_LEN = 'complaint_len'
F_COMPLAINT_CAPS_LEN = 'complaint_caps_len'
F_COMPLAINT_NON_ALPHA_LEN = 'complaint_non_alpha_len'
F_COMPLAINT_EXCLAIMENTION_LEN = 'complaint_exclaimation_len'
F_COMPLAINT_MASK_LEN = 'complaint_mask_len'
# count ?!.
F_COMPLAINT_END_LEN = 'complaint_end_len'
# count digits
F_COMPLAINT_DIGITS_LEN = 'complaint_digit_len'

F_COMPLAINT_CAPS_LEN_NORM = 'complaint_caps_len_norm'
F_COMPLAINT_NON_ALPHA_LEN_NORM = 'complaint_non_alpha_len_norm'
F_COMPLAINT_EXCLAIMENTION_LEN_NORM = 'complaint_exclaimation_len_norm'
F_COMPLAINT_MASK_LEN_NORM = 'complaint_mask_len_norm'
# NORM count ?!.
F_COMPLAINT_END_LEN_NORM = 'complaint_end_len_norm'
# NORM count digit
F_COMPLAINT_DIGITS_LEN_NORM = 'complaint_digit_len_norm'

F_COMPLAINT_TOKENS = 'complaint_toks'
F_MAIN_TOKENS = 'main_tokens'
F_SUB_TOKENS = 'sub_tokens'

LABEL = 'main_subproduct'
DROP = 'drop'

BASE_TRAINING_FEATURES = [F_COMPLAINT_LEN,
                          F_COMPLAINT_CAPS_LEN,
                          F_COMPLAINT_NON_ALPHA_LEN,
                          F_COMPLAINT_EXCLAIMENTION_LEN,
                          F_COMPLAINT_MASK_LEN,
                          F_COMPLAINT_END_LEN,
                          F_COMPLAINT_DIGITS_LEN,
                          F_COMPLAINT_CAPS_LEN_NORM,
                          F_COMPLAINT_NON_ALPHA_LEN_NORM,
                          F_COMPLAINT_EXCLAIMENTION_LEN_NORM,
                          F_COMPLAINT_MASK_LEN_NORM,
                          F_COMPLAINT_END_LEN_NORM,
                          F_COMPLAINT_DIGITS_LEN_NORM]