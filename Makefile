obj :=	main.c \
	net.h \
	net.c \
	mnist.h \
	mnist.c \

default:
	cc $(obj) -o nn -O4 -lm -static

mtrace:
	cc $(obj) -o nn -O4 -lm -static -g -DMTRACE
