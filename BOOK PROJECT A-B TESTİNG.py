import pandas as pd
from scipy.stats import shapiro, levene
from statsmodels.stats.proportion import proportions_ztest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


user_group_1 = pd.read_csv(r'/Users/Cagla/Desktop/user_group_1.csv')    ## türkiye
user_group_2 = pd.read_csv(r'/Users/Cagla/Desktop//user_group_2.csv')   ## hollanda


user_group_1.rename({'Zaman damgası':'Time','Yaşınız nedir?':'Age','Eğitim durumunuz nedir?':'Education_Level','Cinsiyetiniz nedir? ':'Gender',
                     'Bir siteye kaydolurken "Google" veya " Apple" ile kaydolma seçeceği olduğunda bunları kullanmayı tercih eder misiniz?':'Singup_Preferences',
                     'Bildirim arayüzünün kullanıcıya göre ayarlanabilir olması sizin için önemli midir? (Belirli gün veya saatlerde bildirim göndermemesi gibi)':'Notification_Preferences',
                     'Kitap öneri ekranında bulunan kitapların sadece isim bilgisinin olmasını mı yoksa kitaba ait ön bilgilerin olmasını mı tercih edersiniz?':'Recommendation_Preferences'},axis=1,inplace=True)

user_group_1["Education_Level"].replace({'Lise':'Highschool',"Üniversite": "Bachelor", "Yüksek Lisans": "Master",'Diğer':'Other'}, inplace=True)
user_group_1["Gender"].replace({'Kadın':'Female',"Erkek": "Male", "Diğer": "Other"}, inplace=True)
user_group_1["Singup_Preferences"].replace({'Evet':'Yes',"Hayır": "No"}, inplace=True)
user_group_1["Notification_Preferences"].replace({'Evet':'Yes',"Hayır": "No"}, inplace=True)
user_group_1["Recommendation_Preferences"].replace({'Sadece kitabın fotoğrafları':'Only the name of the book',"Kitaba ait ön bilgiler": "Preliminary information of the book"}, inplace=True)
user_group_1["Age"].replace({'35-45':'35-44'}, inplace=True)

user_group_2.rename({'Zaman damgası':'Time','What is your age? ':'Age','What is your education level? ':'Education_Level','What is your gender? ':'Gender',
                     'When registering for a site, would you prefer to use them when there is an option to register with a "Google" and "Apple" account?':'Singup_Preferences',
                     'Is it important for you that the notification interface is user-adjustable? (Like not sending notifications on certain days or hours)':'Notification_Preferences',
                     'Do you prefer the books on the book recommendation screen to have only the name information or to have the preliminary information about the book?':'Recommendation_Preferences'},axis=1,inplace=True)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(user_group_1)
check_df(user_group_2)

user_group_2['Recommendation_Preferences'].fillna('Preliminary information of the book')
user_group_1.drop(['Time'],axis=1,inplace=True)
user_group_2.drop(['Time'],axis=1,inplace=True)

user_group_1.value_counts()
user_group_1.head()
user_group_1["Age"].value_counts()
user_group_1["Education_Level"].value_counts()
user_group_1["Gender"].value_counts()
user_group_1["Singup_Preferences"].value_counts()
user_group_1["Notification_Preferences"].value_counts()
user_group_1["Recommendation_Preferences"].value_counts()

user_group_2.head()
user_group_2["Age"].value_counts()
user_group_2["Education_Level"].value_counts()
user_group_2["Gender"].value_counts()
user_group_2["Singup_Preferences"].value_counts()
user_group_2["Notification_Preferences"].value_counts()
user_group_2["Recommendation_Preferences"].value_counts()


def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols, num_but_cat

cat_cols_1, cat_but_car_1, num_cols_1, num_but_cat_1 = grab_col_names(user_group_1)
cat_cols_2, cat_but_car_2, num_cols_2, num_but_cat_2 = grab_col_names(user_group_2)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

user_group_1 = one_hot_encoder(user_group_1, cat_cols_1, drop_first=True)
user_group_2 = one_hot_encoder(user_group_2, cat_cols_2, drop_first=True)

# Normalisation Assumption
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

for each in user_group_1.columns:
    test_stat, pvalue = shapiro(user_group_1[each])
    print('Test Stat of ', each ,'= %.4f, p-value = %.4f' % (test_stat, pvalue))
## hepsı p<0.05 H0 red normallik varsayımı saglanmamaktadır.


for each in user_group_2.columns:
    test_stat, pvalue = shapiro(user_group_2[each])
    print('Test Stat of ', each ,'= %.4f, p-value = %.4f' % (test_stat, pvalue))
## hepsı p<0.05 H0 red normallik varsayımı saglanmamaktadır.

# Variance Homogenity Assumption
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

for each in user_group_1.columns:
    test_stat, pvalue = levene(user_group_1[each],
                               user_group_2[each])
    print('Test Stat of ', each ,'= %.4f, p-value = %.4f' % (test_stat, pvalue))
### varyans homojenlıgı hepsı ıcın saglanıyor


for each in user_group_2.columns:
    test_stat, pvalue = levene(user_group_2[each],
                               user_group_1[each])
    print('Test Stat of ', each ,'= %.4f, p-value = %.4f' % (test_stat, pvalue))


# Ratio Test
ratio_of_notification_1 = (user_group_1['Notification_Preferences_Yes'].value_counts())[0]/user_group_1['Notification_Preferences_Yes'].count()
ratio_of_notification_2 = (user_group_2['Notification_Preferences_Yes'].value_counts())[1]/user_group_2['Notification_Preferences_Yes'].count()

## H0 : M1=M2  1.GRUPTA BILDIRIM TERCIHLERINE HAYIR DIYENLERIN ORANI ILE 2.GRUPTA BILDIRIM TERCIHLERINE EVET DIYENLERIN ORANI ARASINDA IST.OLARAK ANLAMLI FARK YOKTUR.
## H1 : M1=!M2 1.GRUPTA BILDIRIM TERCIHLERINE EVET DIYENLERIN ORANI ILE 2.GRUPTA BILDIRIM TERCIHLERINE EVET DIYENLERIN ORANI ARASINDA IST.OLARAK ANLAMLI FARK VARDIR.

test_stat, pvalue = proportions_ztest(count=[(user_group_1['Notification_Preferences_Yes'].value_counts())[0], (user_group_2['Notification_Preferences_Yes'].value_counts())[1]],
                                      nobs=[user_group_1['Notification_Preferences_Yes'].count(), user_group_2['Notification_Preferences_Yes'].count()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## Test Stat = -5.4981, p-value = 0.0000  p<0.05 H0 aradakı anlamlı fark vardır.

## H0 : M1=M2 1.GRUPTA KİTAP ÖNERİ EKRANINDA KİTAPLARIN ÖN BİLGİLERİNİN OLMASINI İSTEYENLERİN ORANI İLE
## 2.GRUPTA KİTAP ÖNERİ EKRANINDA KİTAPLARIN SADECE İSİM BİLGİSİNİN OLMASINI İSTEYENLERİN ORANI  ARASINDA IST.OLARAK ANLAMLI FARK YOKTUR.
## H1 : M1=!M2 1.GRUPTA KİTAP ÖNERİ EKRANINDA KİTAPLARIN ÖN BİLGİLERİNİN OLMASINI İSTEYENLERİN ORANI İLE
## 2.GRUPTA KİTAP ÖNERİ EKRANINDA KİTAPLARIN SADECE İSİM BİLGİSİNİN OLMASINI İSTEYENLERİN ORANI ARASINDA IST.OLARAK ANLAMLI FARK VARDIR.

ratio_of_recommendation_1 = (user_group_1['Recommendation_Preferences_Preliminary information of the book'].value_counts())[1]/user_group_1['Recommendation_Preferences_Preliminary information of the book'].count()
ratio_of_recommendation_2 = (user_group_2['Recommendation_Preferences_Preliminary information of the book'].value_counts())[0]/user_group_2['Recommendation_Preferences_Preliminary information of the book'].count()

test_stat, pvalue = proportions_ztest(count=[(user_group_1['Recommendation_Preferences_Preliminary information of the book'].value_counts())[1], (user_group_2['Recommendation_Preferences_Preliminary information of the book'].value_counts())[0]],
                                      nobs=[user_group_1['Recommendation_Preferences_Preliminary information of the book'].count(), user_group_2['Recommendation_Preferences_Preliminary information of the book'].count()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## Test Stat = 4.3187, p-value = 0.0000 p<0.05 h0 reddedilir anlamlı fark vardır.


## H0 : M1=M2 1.GRUPTA BİR SİTEYE KAYDOLURKEN KAYDOLMA SEÇENEĞİ OLDUĞUNDA BUNLARI KULLANANLARIN ORANI İLE
## 2.GRUPTA BİR SİTEYE KAYDOLURKEN KAYDOLMA SEÇENEĞİ OLDUĞUNDA BUNLARI KULLANMAYANLARIN ORANI ARASINDA IST.OLARAK ANLAMLI FARK YOKTUR.
## H1 : M1=!M2 1.GRUPTA KİTAP ÖNERİ EKRANINDA KİTAPLARIN ÖN BİLGİLERİNİN OLMASINI İSTEYENLERİN ORANI İLE
## 2.GRUPTA KİTAP ÖNERİ EKRANINDA KİTAPLARIN ÖN BİLGİLERİNİN OLMASINI İSTEYENLERİN ORANI  ARASINDA IST.OLARAK ANLAMLI FARK VARDIR.


ratio_of_signup_1 = (user_group_1['Singup_Preferences_Yes'].value_counts())[1]/user_group_1['Singup_Preferences_Yes'].count()
ratio_of_signup_2 = (user_group_2['Singup_Preferences_Yes'].value_counts())[0]/user_group_2['Singup_Preferences_Yes'].count()

test_stat, pvalue = proportions_ztest(count=[((user_group_1['Singup_Preferences_Yes'].value_counts()))[1], (user_group_2['Singup_Preferences_Yes'].value_counts())[0]],
                                      nobs=[user_group_1['Singup_Preferences_Yes'].count(), user_group_2['Singup_Preferences_Yes'].count()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## Test Stat = 3.5916, p-value = 0.0003  p<0.05 h0 reddedilir anlamlı fark vardır.