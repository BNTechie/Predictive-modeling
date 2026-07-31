
#Multiple ways to replace 'NA's  with zeros in R dataframe  

#Method 1 - Replace na values with 0 using is.na()
df[is.na(df)] <- 0

#Method 2 - Replace on selected column
df["pages"][is.na(df["<selected column>"])] <- 0
print(df)

#Method 3 - By using replace() & is.na()
df <- replace(df, is.na(df), 0)

#Method 4 
df <- df %>% replace(is.na(.), 0)

#Method 5 - with the imputeTS package
library("imputeTS")
#Replace NA avalues with 0
df <- na_replace(df, 0)

#Method 6 - Replace NA with zero on all numeric column
library("dplyr")
df <- mutate_all(df, ~coalesce(.,0))

#With these two libraries
library("tidyr")
library("dplyr")

#Method 7 - Replace NA with zero on all numeric column
df <- mutate_all(df, ~replace_na(.,0))

#Method 8 - Replace NA using setnafill() from data.table
library("data.table")
df <- setnafill(df, fill=0)

#Method 9 - Replace na with zero on specific numeric column
#Load dplyr library
df <- df %>% 
  mutate(id = coalesce(id, 0))

# Method 10 - Replace on multiple columns
df <- df %>% 
  mutate(id = coalesce(id, 0),
         pages = coalesce("<selected column>", 0))

# Method 11 - with tidyr library
df <- df %>% 
  mutate_at(1, ~replace_na(.,0))

# Method 12 - Replace NA on multiple columns by Index
df <- df %>% 
  mutate_at(c(1,3), ~replace_na(.,0))

# Method 13 - Replace NA on multiple columns by name
df <- df %>% 
  mutate_at(c('id','<selected column>'), ~replace_na(.,0))

# Method 14 - Replace only numeric columns
df <- df %>% 
  mutate_if(is.numeric, ~replace_na(., 0))
