obj :=	main.c \
	net.h \
	net.c \
	mnist.h \
	mnist.c \

default:
	gcc $(obj) -o nn -O4 -lm -static

# for memory debug
mtrace:
	gcc $(obj) -o nn -O4 -lm -g
	./nn
	mtrace nn mem.log
