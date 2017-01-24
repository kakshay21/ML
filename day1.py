import theano
import theano.tensor as t 
a = t.dscalar()
b = t.dscalar()
c = a + b
f = theano.function([a,b], c)
result= f(1.5, 2.5)
print(result)