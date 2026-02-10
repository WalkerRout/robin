// public contract between compilation stages, consumes input to avoid dangling state
pub trait Pass<I>
where
  I: ?Sized,
{
  type Output;
  type Error;
  fn run(self, input: &I) -> Result<Self::Output, Self::Error>;
}

// blanket trait providing `.accept(pass)` sugar
pub trait Acceptor {
  fn accept<P>(&self, pass: P) -> Result<P::Output, P::Error>
  where
    P: Pass<Self>,
  {
    pass.run(self)
  }
}

impl<T> Acceptor for T where T: ?Sized {}
