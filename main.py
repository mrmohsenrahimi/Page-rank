import numpy  # وارد کردن کتابخونه NumPy برای کار با ماتریس‌ ها و محاسبات عددی


def pagerank(input_matrix):  # تعریف تابع pagerank که ماتریس مجاورت ۴×۴ رو می‌گیره

    row_count = input_matrix.shape[0]  # تشخیص تعداد سطر ماتریس

    # محاسبه مجموع ستون های هر سطر
    row_sums = input_matrix.sum(axis=1)  # axis=1 یعنی جمع ستون‌ها برای هر سطر، نتیجه یه آرایه ۴ تاییه

    # ساخت یه ماتریس خالی ۴×۴ (مثل input_matrix) برای ذخیره ماتریس انتقال، با نوع داده اعشاری
    transition_matrix = numpy.zeros_like(input_matrix, dtype=float) # create an empty matrix

    # حلقه برای پردازش هر سطر ماتریس (هر صفحه)
    for i in range(row_count):  # برای هر کدوم از ۴ صفحه (i از ۰ تا ۳)
        if row_sums[i] == 0:  # اگه صفحه هیچ لینک خروجی نداشته باشه
            transition_matrix[i, :] = 1.0 / row_count  # به هر صفحه احتمال یکنواخت (۱/۴) بده
        else:  # اگه صفحه لینک خروجی داره
            transition_matrix[i, :] = input_matrix[i, :] / row_sums[i]  # هر عدد یک سطر رو تقسیم بر جمع سطر میکنیم تا احتمال‌ها به دست بیاد

    pagerank_output = numpy.ones((row_count, 1)) / row_count # ساخت یک ماتریس بعنوان خروجی اولیه الگوریتم پیج رنک که همه سطر ها (احتمال ها) برابر یک چهارم (۱/۴) هست

    print("خروجی اولیه الگوریتم page rank")
    print(pagerank_output)  # چاپ خروجی اولیه PageRank همه یک چهارم

    for iteration in range(4):
        # ضرب ماتریس انتقال (ترانهاده) در بردار PageRank فعلی برای پخش امتیازها
        pagerank_output = numpy.dot(transition_matrix.T, pagerank_output)  # .T ماتریس رو ترانهاده می‌کنه، numpy.dot ضرب ماتریسیه

        # نرمال‌سازی بردار PageRank تا جمع امتیازها ۱ بشه
        pagerank_output = pagerank_output / pagerank_output.sum()  # تقسیم بر جمع کل خروجی pagerank برای نگه داشتن مجموع احتمالات روی عدد ۱

        # چاپ پیام برای نشون دادن شماره مرحله
        print(f"\nخروجی الگوریتم page rank در پیمایش {iteration + 1}:")  # iteration + 1 برای نمایش مرحله ۱ تا ۴
        print(pagerank_output)  # چاپ بردار PageRank بعد از این مرحله

    return pagerank_output  # برای استفاده در پرینت نهایی


if __name__ == "__main__":

    input_matrix = numpy.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])

    print("ماتریس اولیه")
    print(input_matrix)

    # صدا زدن تابع pagerank برای محاسبه PageRank
    final_pr = pagerank(input_matrix)