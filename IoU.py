import math

eye1= [316,162,33,319,163,106]
eye2= [155,117,24,155,117,56]
eye3= [234,230,56,248,232,221]
Ax=575
Ay= 73
ra= 109
Bx=319
By=163
rb=106

a = Bx - Ax
b = By - Ay
c = math.pow(a,2)

e = math.pow(b,2)

f = c+e
d = math.sqrt(f)
x = ra - rb
y = ra + rb
print(y)
print(x)
print(d)

if y <= d:
    print('False positive found')
    print(' cIoU = 0')

h = math.pow(ra,2) - math.pow(rb,2) + math.pow(d,2)

Lx = h/(2*d)

j = math.pow(ra,2) - math.pow(Lx,2)

Ly = math.sqrt(j)
sums = math.pow(ra,2) + math.pow(rb,2) + math.pow(Lx, 2)


d1 = (math.pow(ra,2) - math.pow(rb,2) + math.pow(d,2))/(2*d)
d2 = d-d1

l = d1/ra
s = d2/rb
u=math.pow(ra,2) - math.pow(d1,2)
k= math.pow(rb,2) - math.pow(d2,2)

AprienikB = math.pow(ra,2)*math.acos(l) - d1*math.sqrt(u) + math.pow(rb,2)*math.acos(s) - d2*math.sqrt(k)
AzjednotB = math.pi * math.pow(ra,2) + math.pi * math.pow(rb,2) - AprienikB

vysledok = AprienikB/AzjednotB

print(vysledok)