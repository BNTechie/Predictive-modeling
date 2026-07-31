# Method 1 - Using empty data.frame()
df = data.frame()

# Method 2 - Using matrix()
df = data.frame(matrix(nrow = 0, ncol = 0)) 

# Method 3 - Assign column names to empty DataFrame
# Create an Empty DataFrame with 0 rows and n columns 
columns = c("id","name","dob") 
df = data.frame(matrix(nrow = 0, ncol = length(columns))) 
colnames(df) = columns

# Method 4
# Assign columns & datatype to empty DataFrame
df = data.frame(id=numeric(0),name=character(0),dob=character(0))

# Method 5
# By Setting column variables with NA
df = data.frame(id=NA, name=NA, dob=NA)[numeric(0), ]

# Method 6
# Create empty DataFrame from existing
empty_df = df[FALSE,]



# Create an Empty DataFrame
df = data.frame()

#Create an Empty DataFrame with 0 rows and 0 columns
matrix_empty = matrix(nrow = 0, ncol = 0)
df = data.frame(matrix_empty) 

# Create a Vector with Columns
columns = c("id","name","dob") 

#Create a Empty DataFrame with 0 rows and n columns
df = data.frame(matrix(nrow = 0, ncol = length(columns))) 

# Assign column names
colnames(df) = columns

# Using empty vectors
df2 = data.frame(id=numeric(0),name=character(0),dob=character(0))



# Create Empty DataFrame with column names
df3 = data.frame(id=NA, name=NA, dob=NA)[numeric(0), ]
print(df3)


# Create a DataFrame
df <- data.frame(
  id=c(10,11,12,13),
  name=c('sai','ram','deepika','sahithi'),
  dob=as.Date(c('1990-10-02','1981-3-24','1987-6-14','1985-8-16'))
)

# Create empty DataFrame with columns from existing DataFrame.
empty_df = df[FALSE,]


###https://sparkbyexamples.com/r-programming/r-create-an-empty-dataframe/




