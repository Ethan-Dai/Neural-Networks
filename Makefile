obj :=	main.c \
	net.h \
	net.c \
	mnist.h \
	mnist.c \

default:
	gcc $(obj) -o nn -O4 -lm
